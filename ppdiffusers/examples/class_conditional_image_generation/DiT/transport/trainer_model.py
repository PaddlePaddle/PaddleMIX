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
import os
import json
import paddle
import paddle.nn as nn

from ppdiffusers import AutoencoderKL, is_ppxformers_available
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params
from paddlenlp.utils.log import logger

from .sit import SiT
from .transport import ModelType, WeightType, PathType
from . import path
from .utils import mean_flat


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class SiTDiffusionModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vqvae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")

        self.model_mean_type = "epsilon" # PREVIOUS_X START_X EPSILON
        self.model_var_type = "learned_range" # LEARNED FIXED_SMALL FIXED_LARGE LEARNED_RANGE
        self.loss_type = "mse" # MSE RESCALED_MSE KL(is_vb) RESCALED_KL(is_vb)

        self.path_type = "Linear" # LINEAR GVP VP
        self.prediction = "velocity" # VELOCITY NOISE SCORE 
        self.model_type = "velocity" #
        self.loss_weight = "None" #
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


        self.transformer = SiT(**read_json(model_args.config_file))
        self.transformer_is_pretrained = False

        assert model_args.prediction_type in ["epsilon", "v_prediction"]
        self.prediction_type = model_args.prediction_type

        self.use_ema = model_args.use_ema
        self.noise_offset = model_args.noise_offset
        if self.use_ema:
            self.model_ema = LitEma(self.transformer)
        self.transformer.train()
        self.vae.eval()

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
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

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
        return loss

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, max_batch=8, **kwargs):
        self.eval()
        if pixel_values.shape[0] > max_batch:
            pixel_values = pixel_values[:max_batch]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

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
                    logger.warn(
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
