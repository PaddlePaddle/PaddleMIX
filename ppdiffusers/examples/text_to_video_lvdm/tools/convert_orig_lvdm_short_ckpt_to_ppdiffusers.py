# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import pickle
from functools import lru_cache

import numpy as np
import paddle
import torch

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the SD checkpoints. Please install it with `pip install OmegaConf`."
    )

from ppdiffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    LVDMAutoencoderKL,
    LVDMUncondPipeline,
    LVDMUNet3DModel,
    PNDMScheduler,
)

paddle.set_device("cpu")
MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30


class TensorMeta:
    """
    metadata of tensor
    """

    def __init__(self, key: str, n_bytes: int, dtype: str):
        self.key = key
        self.nbytes = n_bytes
        self.dtype = dtype
        self.size = None

    def __repr__(self):
        return f"size: {self.size} key: {self.key}, nbytes: {self.nbytes}, dtype: {self.dtype}"


@lru_cache(maxsize=None)
def _storage_type_to_dtype_to_map():
    """convert storage type to numpy dtype"""
    return {
        "DoubleStorage": np.double,
        "FloatStorage": np.float32,
        "HalfStorage": np.half,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool8,
        "ComplexDoubleStorage": np.cdouble,
        "ComplexFloatStorage": np.cfloat,
    }


class StorageType:
    """Temp Class for Storage Type"""

    def __init__(self, name):
        self.dtype = _storage_type_to_dtype_to_map()[name]

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _element_size(dtype: str) -> int:
    """
    Returns the element size for a dtype, in bytes
    """
    if dtype in [np.float16, np.float32, np.float64]:
        return np.finfo(dtype).bits >> 3
    elif dtype == np.bool8:
        return 1
    else:
        return np.iinfo(dtype).bits >> 3


class UnpicklerWrapperStage(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return StorageType(name)
            except KeyError:
                pass

        # pure torch tensor builder
        if mod_name == "torch._utils":
            return _rebuild_tensor_stage

        # pytorch_lightning tensor builder
        if mod_name == "pytorch_lightning":
            return dumpy
        return super().find_class(mod_name, name)


def get_data_iostream(file: str, file_name="data.pkl"):
    FILENAME = f"archive/{file_name}".encode("latin")
    padding_size_plus_fbxx = 4 + 14
    data_iostream = []
    offset = MZ_ZIP_LOCAL_DIR_HEADER_SIZE + len(FILENAME) + padding_size_plus_fbxx
    with open(file, "rb") as r:
        r.seek(offset)
        for bytes_data in io.BytesIO(r.read()):
            if b".PK" in bytes_data:
                data_iostream.append(bytes_data.split(b".PK")[0])
                data_iostream.append(b".")
                break
            data_iostream.append(bytes_data)
    out = b"".join(data_iostream)
    return out, offset + len(out)


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if isinstance(storage, TensorMeta):
        storage.size = size
    return storage


def dumpy(*args, **kwarsg):
    return None


def create_unet_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params

    config = dict(
        image_size=unet_params.image_size,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        model_channels=unet_params.model_channels,
        attention_resolutions=unet_params.attention_resolutions,
        num_res_blocks=unet_params.num_res_blocks,
        channel_mult=unet_params.channel_mult,
        num_heads=unet_params.num_heads,
        use_temporal_transformer=unet_params.use_temporal_transformer,
        legacy=unet_params.legacy,
        kernel_size_t=unet_params.kernel_size_t,
        padding_t=unet_params.padding_t,
        temporal_length=unet_params.temporal_length,
        use_relative_position=unet_params.use_relative_position,
        use_scale_shift_norm=unet_params.use_scale_shift_norm,
    )

    return config


def create_lvdm_vae_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim
    config = dict(
        n_hiddens=vae_params.encoder.params.n_hiddens,
        downsample=vae_params.encoder.params.downsample,
        image_channel=vae_params.encoder.params.image_channel,
        norm_type=vae_params.encoder.params.norm_type,
        padding_type=vae_params.encoder.params.padding_type,
        double_z=vae_params.encoder.params.double_z,
        z_channels=vae_params.encoder.params.z_channels,
        upsample=vae_params.decoder.params.upsample,
    )
    return config


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config.model.params.timesteps,
        beta_start=original_config.model.params.linear_start,
        beta_end=original_config.model.params.linear_end,
        beta_schedule="scaled_linear",
    )
    return schedular


def convert_lvdm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())

    unet_key = "model.diffusion_model."
    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        if extract_ema:
            print(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
        else:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )

    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = unet_state_dict

    return new_checkpoint


def convert_lvdm_vae_checkpoint(checkpoint, vae_checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    if vae_checkpoint:
        vae_state_dict = vae_checkpoint
    else:
        vae_key = "first_stage_model."
        keys = list(checkpoint.keys())
        for key in keys:
            if key.startswith(vae_key):
                vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = vae_state_dict
    return new_checkpoint


def convert_diffusers_vae_unet_to_ppdiffusers(vae_or_unet, diffusers_vae_unet_checkpoint, dtype="float32"):
    need_transpose = []
    for k, v in vae_or_unet.named_sublayers(include_self=True):
        if isinstance(v, paddle.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = {}
    for k, v in diffusers_vae_unet_checkpoint.items():
        if k not in need_transpose:
            new_vae_or_unet[k] = v.numpy().astype(dtype)
        else:
            new_vae_or_unet[k] = v.t().numpy().astype(dtype)
    return new_vae_or_unet


def check_keys(model, state_dict):
    cls_name = model.__class__.__name__
    missing_keys = []
    mismatched_keys = []
    for k, v in model.state_dict().items():
        if k not in state_dict.keys():
            missing_keys.append(k)
        if list(v.shape) != list(state_dict[k].shape):
            mismatched_keys.append(str((k, list(v.shape), list(state_dict[k].shape))))
    if len(missing_keys):
        missing_keys_str = ", ".join(missing_keys)
        print(f"{cls_name} Found missing_keys {missing_keys_str}!")
    if len(mismatched_keys):
        mismatched_keys_str = ", ".join(mismatched_keys)
        print(f"{cls_name} Found mismatched_keys {mismatched_keys_str}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output model.",
    )
    args = parser.parse_args()

    # image_size = 512
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    checkpoint = checkpoint.get("state_dict", checkpoint)

    vae_checkpoint = None
    if args.vae_checkpoint_path:
        vae_checkpoint = torch.load(args.vae_checkpoint_path, map_location="cpu")
        vae_checkpoint = vae_checkpoint.get("state_dict", vae_checkpoint)

    original_config = OmegaConf.load(args.original_config_file)

    if args.num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = args.num_in_channels

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
    )

    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if args.scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif args.scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {args.scheduler_type} doesn't exist!")

    # 1. Convert the LVDMUNet3DModel model.
    diffusers_unet_config = create_unet_diffusers_config(original_config)
    diffusers_unet_checkpoint = convert_lvdm_unet_checkpoint(
        checkpoint,
        diffusers_unet_config,
        path=args.checkpoint_path,
        extract_ema=args.extract_ema,
    )
    unet = LVDMUNet3DModel.from_config(diffusers_unet_config)
    ppdiffusers_unet_checkpoint = convert_diffusers_vae_unet_to_ppdiffusers(unet, diffusers_unet_checkpoint)
    check_keys(unet, ppdiffusers_unet_checkpoint)
    unet.load_dict(ppdiffusers_unet_checkpoint)

    # 2. Convert the LVDMAutoencoderKL model.
    vae_config = create_lvdm_vae_diffusers_config(original_config)
    diffusers_vae_checkpoint = convert_lvdm_vae_checkpoint(checkpoint, vae_checkpoint, vae_config)
    vae = LVDMAutoencoderKL.from_config(vae_config)
    ppdiffusers_vae_checkpoint = convert_diffusers_vae_unet_to_ppdiffusers(vae, diffusers_vae_checkpoint)
    check_keys(vae, ppdiffusers_vae_checkpoint)
    vae.load_dict(ppdiffusers_vae_checkpoint)

    pipe = LVDMUncondPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
    )

    pipe.save_pretrained(args.dump_path)
