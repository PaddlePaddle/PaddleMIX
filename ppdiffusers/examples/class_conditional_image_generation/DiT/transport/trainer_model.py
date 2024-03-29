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

import contextlib
import json
import os

import paddle
import paddle.nn as nn
from paddlenlp.utils.log import logger

from ppdiffusers import AutoencoderKL, is_ppxformers_available
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params

from . import path
from .sit import SiT
from .transport import ModelType, PathType, WeightType
from .utils import mean_flat


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_model(args, model, optimizer=None, ckpt_dir=""):
    """
    load the saved checkpoint file and update the state dicts of model and optimizer.
    """
    if ckpt_dir and isinstance(ckpt_dir, str) and os.path.isdir(ckpt_dir):
        print("Try to load checkpoint from %s " % ckpt_dir)

        load_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}".format(ckpt_dir, args.mp_rank, args.sharding_rank)
        model_path = os.path.join(load_dir, "model.pdparams")
        opt_path = os.path.join(load_dir, "model_state.pdopt")
        # meta_path = os.path.join(load_dir, "meta_state.pdopt")

        if os.path.exists(model_path):
            model_dict = paddle.load(model_path)
            for name, param in model.state_dict().items():
                assert name in model_dict.keys(), "No param named `{}` was found in checkpoint file.".format(name)

                if param.dtype != model_dict[name].dtype:
                    model_dict[name] = model_dict[name].cast(param.dtype)

            model.set_state_dict(model_dict)
            del model_dict
        else:
            raise ValueError("No checkpoint file found in %s" % model_path)

        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
            del opt_dict
        else:
            print("No optimizer checkpoint file found in %s." % opt_path)

        # if os.path.exists(meta_path):
        #     meta_dict = paddle.load(meta_path)
        #     load_recovery = {
        #         'step': meta_dict['step'],
        #         'epoch': meta_dict['epoch'],
        #         'rng_state': meta_dict['cuda_rng_state']
        #     }
        #     del meta_dict
        # else:
        #     raise ValueError("No meta checkpoint file found in %s." %
        #                         meta_path)

        print("successfully load checkpoints")

    elif ckpt_dir and os.path.isfile(ckpt_dir):
        print("Try to load a whole checkpoint from %s " % ckpt_dir)
        embedding_list = ["embedding_table"]
        collinear_list = [
            "wq",
            "wk",
            "wv",
            "adaLN_modulation",  # TransformerBlock ParallelFinalLayer
            "linear",
            "x_embedder",
            "w1",
            "w3",
            "fc1",
            "fc2",
            "qkv",
            "proj",
            "mlp",  # ParallelTimestepEmbedder
            "q_norm",
            "k_norm",
            # "attention_norm",
            # "ffn_norm",
        ]
        rowlinear_list = ["wo", "w2"]
        all_list = collinear_list + rowlinear_list + embedding_list
        skip_list = [
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
        ]

        col_list = []
        row_list = []
        emb_list = []

        mp_rank = args.mp_rank
        mp_size = max(args.tensor_parallel_degree, 1)

        def col_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[1] // mp_size
                return model_dict[:, mp_rank * subbatch : (mp_rank + 1) * subbatch]
            elif len(model_dict.shape) == 1:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

        def row_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]
            else:
                return model_dict

        def emb_split_modeldict(model_dict):
            subbatch = model_dict.shape[0] // mp_size
            return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

        model_dict = paddle.load(ckpt_dir)
        modelkeys = model_dict.keys()
        for whole_key in modelkeys:
            if "." not in whole_key:
                continue

            if "mlp" in whole_key or "adaLN_modulation" in whole_key:
                # TODO: remove nn.Sequential
                # self.mlp = nn.Sequential(linear(), silu(), linear())
                # self.adaLN_modulation = nn.Sequential(silu(), linear())
                key = whole_key.split(".")[-3]
            else:
                key = whole_key.split(".")[-2]

            if whole_key in skip_list:
                continue
            if key in all_list:
                if key in collinear_list:
                    col_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = col_split_modeldict(model_dict[whole_key])
                elif key in rowlinear_list:
                    row_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = row_split_modeldict(model_dict[whole_key])
                else:
                    # embedding_list
                    emb_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = emb_split_modeldict(model_dict[whole_key])
            # else:
            #     print(f'bad whole_key:{whole_key}, the key:{key} is not in [col + rowl + embed] list')
            #     exit()

        print("cast state_dict to default dtype:{}".format(paddle.get_default_dtype()))
        model.set_state_dict(model_dict)
        del model_dict
    else:
        print("`load` requires a valid value of `ckpt_dir`.")
        raise TypeError("`load` requires a valid value of `ckpt_dir`.")


class SiTDiffusionModel(nn.Layer):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args

        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vqvae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)

        # init SiT
        if model_args.pretrained_model_name_or_path is None:
            self.transformer = SiT(**read_json(model_args.config_file))
            # Note: Initialize SiT in transport/sit.py
            logger.info("Init SiT model from scratch!")
        else:
            self.transformer = SiT.from_pretrained(model_args.pretrained_model_name_or_path, subfolder="transformer")
            logger.info(f"Init SiT model from {model_args.pretrained_model_name_or_path}!")

        # make sure unet in train mode, vae and text_encoder in eval mode
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")
        self.vae.eval()
        self.transformer.train()

        self.use_ema = False
        self.model_ema = None
        if self.use_ema:
            self.model_ema = LitEma(self.transformer)

        if model_args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
            try:
                self.transformer.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(
                    "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                    f" correctly and a GPU is available: {e}"
                )

        # other settings
        self.model_mean_type = "epsilon"  # PREVIOUS_X START_X EPSILON
        self.model_var_type = "learned_range"  # LEARNED FIXED_SMALL FIXED_LARGE LEARNED_RANGE
        self.loss_type = "mse"  # MSE RESCALED_MSE KL(is_vb) RESCALED_KL(is_vb)
        self.path_type = "Linear"  # LINEAR GVP VP
        self.prediction = "velocity"  # VELOCITY NOISE SCORE
        self.model_type = "velocity"  #
        self.loss_weight = "None"  #
        self.train_eps = 0
        self.sample_eps = 0

        path_choice = {
            "Linear": PathType.LINEAR,
            "GVP": PathType.GVP,
            "VP": PathType.VP,
        }
        path_type = path_choice[self.path_type]

        if self.loss_weight == "velocity":
            loss_type = WeightType.VELOCITY
        elif self.loss_weight == "likelihood":
            loss_type = WeightType.LIKELIHOOD
        else:
            loss_type = WeightType.NONE
        self.loss_type = loss_type

        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.path_sampler = path_options[path_type]()

        assert model_args.prediction_type in ["epsilon", "v_prediction"]
        self.prediction_type = model_args.prediction_type

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.transformer.parameters())
            self.model_ema.copy_to(self.transformer)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.transformer)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """

        x0 = paddle.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = paddle.rand((x1.shape[0],)) * (t1 - t0) + t0
        return t, x0, x1

    def forward(self, latents=None, label_id=None, **kwargs):
        t, x0, x1 = self.sample(latents)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        self.vae.eval()
        model_output = self.transformer(x=xt, t=t, y=label_id)
        B, *_, C = xt.shape
        assert model_output.shape == [B, *xt.shape[1:-1], C]

        if self.model_type == "velocity":
            loss = mean_flat(((model_output - ut) ** 2))
        else:
            raise NotImplementedError()
        return loss.mean()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.transformer.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self):
        self.set_recompute(True)

    def set_xformers(self, use_xformers=False):
        if use_xformers:
            if not is_ppxformers_available():
                raise ValueError(
                    'Please run `python -m pip install "paddlepaddle-gpu>=2.5.0.post117" -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html first.'
                )
            else:
                try:
                    attention_op = os.getenv("FLAG_XFORMERS_ATTENTION_OP", "none").lower()

                    if attention_op == "none":
                        attention_op = None

                    self.transformer.enable_xformers_memory_efficient_attention(attention_op)
                    if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
                        self.vae.enable_xformers_memory_efficient_attention(attention_op)
                except Exception as e:
                    logger.warning(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )
        else:
            if hasattr(self.transformer, "set_default_attn_processor"):
                self.transformer.set_default_attn_processor()
            if hasattr(self.vae, "set_default_attn_processor"):
                self.vae.set_default_attn_processor()

    def set_ema(self, use_ema=False):
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.transformer)
