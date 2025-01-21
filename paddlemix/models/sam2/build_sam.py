# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import paddle
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": ("configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "facebook/sam2-hiera-small": ("configs/sam2/sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "facebook/sam2-hiera-base-plus": ("configs/sam2/sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "facebook/sam2-hiera-large": ("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    "facebook/sam2.1-hiera-tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "facebook/sam2.1-hiera-small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "facebook/sam2.1-hiera-base-plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "facebook/sam2.1-hiera-large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def build_sam2(
    config_file, ckpt_path=None, mode="eval", hydra_overrides_extra=[], apply_postprocessing=True, **kwargs
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)

    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file, ckpt_path=None, mode="eval", hydra_overrides_extra=[], apply_postprocessing=True, **kwargs
):
    hydra_overrides = ["++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor"]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)
    cfg = compose(config_name=config_file, overrides=hydra_overrides)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)

    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = paddle.load(path=ckpt_path)
        missing_keys, unexpected_keys = model.set_state_dict(state_dict=sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
