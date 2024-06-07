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
from math import ceil
from typing import List

import config
import numpy as np
import paddle
import PIL
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.initializer import TruncatedNormal
from tqdm import tqdm

from ppdiffusers.initializer import reset_initialized_parameter, zeros_
from ppdiffusers.models.attention_processor import Attention
from ppdiffusers.models.ema import LitEma
from ppdiffusers.models.stable_cascade.gdf import (
    GDF,
    AdaptiveLossWeight,
    CosineSchedule,
    CosineTNoiseCond,
    DDPMSampler,
    EpsilonTarget,
    P2LossWeight,
    VPScaler,
)
from ppdiffusers.models.stable_cascade.modules.effnet import EfficientNetEncoder
from ppdiffusers.models.stable_cascade.modules.stage_a import StageA
from ppdiffusers.models.stable_cascade.modules.stage_b import StageB
from ppdiffusers.transformers import AutoTokenizer, CLIPTextModelWithProjection

from .core import EXPECTED, load_or_fail

trunc_normal_ = TruncatedNormal(std=0.02)


def load():
    return paddle.to_tensor(np.load("/root/lxl/0_SC/work/baidu/personal-code/stable_cascade/work/pred.npy"))


def diff(a: paddle.Tensor, b: paddle.Tensor) -> float:
    return (a - b).abs().mean().item()


def calculate_latent_sizes(
    height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0
):
    latent_height = ceil(height / compression_factor_b)
    latent_width = ceil(width / compression_factor_b)
    stage_c_latent_shape = batch_size, 16, latent_height, latent_width
    latent_height = ceil(height / compression_factor_a)
    latent_width = ceil(width / compression_factor_a)
    stage_b_latent_shape = batch_size, 4, latent_height, latent_width
    return stage_c_latent_shape, stage_b_latent_shape


class ModelB(nn.Layer):
    class Config:
        dtype: str = config.dtype
        training: bool = False
        use_ema: bool = False
        model_version: str = config.model_b_version
        clip_text_model_name: str = config.clip_text_model_name
        effnet_checkpoint_path: str = config.effnet_checkpoint_path
        stage_a_checkpoint_path: str = config.stage_a_checkpoint_path
        unet_checkpoint_path: str = config.stage_b_checkpoint_path
        shift: float = None
        adaptive_loss_weight: bool = False
        grad_accum_steps: int = 1

    class Extras:
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: paddle.vision.transforms.Compose = EXPECTED
        clip_preprocess: paddle.vision.transforms.Compose = EXPECTED

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Models:
        tokenizer: paddle.nn.Layer = EXPECTED
        text_model: paddle.nn.Layer = EXPECTED
        image_model: paddle.nn.Layer = None
        stage_a: paddle.nn.Layer = EXPECTED
        unet: paddle.nn.Layer = None
        effnet: paddle.nn.Layer = EXPECTED

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def setup_extras_pre(self) -> Extras:
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(),
            target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight() if self.config.adaptive_loss_weight is True else P2LossWeight(),
        )
        sampling_configs = {
            "cfg": 1.5,
            "sampler": DDPMSampler(gdf),
            "shift": 1,
            "timesteps": 10,
        }

        effnet_preprocess = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=None,
        )

    def load_models(self, skip_clip: bool = False):
        dtype = self.config.dtype if self.config.dtype else "float32"

        # load effnet
        effnet = EfficientNetEncoder()
        effnet_checkpoint = load_or_fail(self.config.effnet_checkpoint_path)
        effnet.set_state_dict(
            effnet_checkpoint["state_dict"] if "state_dict" in effnet_checkpoint else effnet_checkpoint
        )
        effnet.stop_gradient = True
        effnet.eval()
        effnet.to(dtype=dtype)
        del effnet_checkpoint

        # load previewer
        stage_a = StageA()
        stage_a_checkpoint = load_or_fail(self.config.stage_a_checkpoint_path)
        stage_a.set_state_dict(
            state_dict=stage_a_checkpoint
            if "state_dict" not in stage_a_checkpoint
            else stage_a_checkpoint["state_dict"]
        )
        stage_a.eval()
        stage_a.requires_grad = False
        del stage_a_checkpoint

        # load unet
        if self.config.model_version == "3B":
            self.unet = StageB(
                c_hidden=[320, 640, 1280, 1280],
                nhead=[-1, -1, 20, 20],
                blocks=[[2, 6, 28, 6], [6, 28, 6, 2]],
                block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]],
            )
        elif self.config.model_version == "700M":
            self.unet = StageB(
                c_hidden=[320, 576, 1152, 1152],
                nhead=[-1, 9, 18, 18],
                blocks=[[2, 4, 14, 4], [4, 14, 4, 2]],
                block_repeat=[[1, 1, 1, 1], [2, 2, 2, 2]],
            )
        else:
            raise ValueError(f"Unknown model version {self.config.model_version}")
        if self.config.unet_checkpoint_path is not None:
            self.unet.set_state_dict(load_or_fail(self.config.unet_checkpoint_path))
        self.unet.to(dtype=dtype)

        if skip_clip:
            tokenizer = None
            text_model = None
        else:
            # load tokenizer, text_model, image_model
            tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
            text_model = CLIPTextModelWithProjection.from_pretrained(self.config.clip_text_model_name)
            text_model.eval()
            # text_model.to(dtype=self.config.dtype)

        # set models
        self.models = self.Models(
            tokenizer=tokenizer,
            text_model=text_model,
            stage_a=stage_a,
            unet=self.unet,
            effnet=effnet,
        )

    def __init__(self, model_args=None, train_args=None):
        super().__init__()
        # 初始化配置，包括模型参数、训练参数等
        self.config = self.Config()
        if model_args is not None:
            self.config.training = model_args.training
        if train_args is not None:
            self.config.grad_accum_steps = train_args.gradient_accumulation_steps

        # 加载预处理信息
        self.extras = self.setup_extras_pre()

        # 加载预训练模型
        self.load_models()

    @paddle.no_grad()
    def get_conditions(
        self,
        batch: dict,
        models: Models,
        extras: Extras,
        is_eval=False,
        is_unconditional=False,
        eval_image_embeds=False,
        return_fields=None,
    ):
        if return_fields is None:
            return_fields = ["clip_text", "clip_text_pooled", "clip_img"]
        captions = batch.get("captions", None)
        images = batch.get("images", None)
        if images is not None:
            # images = images.to(self.device)
            if is_eval and not is_unconditional:
                effnet_embeddings = models.effnet(extras.effnet_preprocess(images))
            else:
                if is_eval:
                    effnet_factor = 1
                else:
                    effnet_factor = np.random.uniform(0.5, 1)
                effnet_height, effnet_width = int(images.shape[-2] * effnet_factor // 32 * 32), int(
                    images.shape[-1] * effnet_factor // 32 * 32
                )
                effnet_embeddings = paddle.zeros(shape=[images.shape[0], 16, effnet_height // 32, effnet_width // 32])
                if not is_eval:
                    if not is_eval:
                        effnet_images = paddle.nn.functional.interpolate(
                            images, [effnet_height, effnet_width], mode="NEAREST"
                        )
                    rand_idx = np.random.rand(len(images)) <= 0.9
                    if any(rand_idx):
                        effnet_embeddings[rand_idx] = models.effnet(extras.effnet_preprocess(effnet_images[rand_idx]))
        else:
            effnet_embeddings = None
        images = paddle.to_tensor(images) if images else None
        batch_size = len(captions)
        text_pooled_embeddings = None
        if "clip_text" in return_fields or "clip_text_pooled" in return_fields:
            if is_eval:
                if is_unconditional:
                    captions_unpooled = ["" for _ in range(batch_size)]
                else:
                    captions_unpooled = captions
            else:
                rand_idx = np.random.rand(batch_size) > 0.05
                captions_unpooled = [(str(c) if keep else "") for c, keep in zip(captions, rand_idx)]
            clip_tokens_unpooled = models.tokenizer(
                captions_unpooled,
                truncation=True,
                padding="max_length",
                max_length=models.tokenizer.model_max_length,
                return_tensors="pd",
            )

            with paddle.no_grad():
                text_encoder_output = models.text_model(**clip_tokens_unpooled, output_hidden_states=True)
            if "clip_text_pooled" in return_fields:
                text_pooled_embeddings = text_encoder_output.text_embeds.unsqueeze(1)
        image_embeddings = None
        if "clip_img" in return_fields:
            image_embeddings = paddle.zeros(shape=[batch_size, 768])  # 全零的图像特征
            if images is not None:
                if is_eval:
                    # 在测试阶段，直接使用真实图像特征
                    if not is_unconditional and eval_image_embeds:
                        image_embeddings = models.image_model(extras.clip_preprocess(images)).image_embeds
                else:
                    pass
                    # 在训练阶段，随机采样部分图像特征，其余全零
                    rand_idx = np.random.rand(batch_size) > 0.9
                    if np.any(rand_idx):
                        for i, img in enumerate(images):
                            if rand_idx[i]:
                                image_embeddings[i] = models.image_model(
                                    extras.clip_preprocess(img).unsqueeze(0)
                                ).image_embeds[0]
            image_embeddings = image_embeddings.unsqueeze(axis=1)
        return {
            "effnet": effnet_embeddings,
            "clip": text_pooled_embeddings,
        }

    def init_unet_weights(self):
        reset_initialized_parameter(self.unet)
        for _, m in self.unet.named_sublayers():
            if isinstance(m, Attention) and getattr(m, "group_norm", None) is not None:
                zeros_(m.to_out[0].weight)
                zeros_(m.to_out[0].bias)
            if isinstance(
                m, (nn.Linear, fleet.meta_parallel.ColumnParallelLinear, fleet.meta_parallel.RowParallelLinear)
            ):
                trunc_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    zeros_(m.bias)

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.config.use_ema:
            self.model_ema.store(self.unet.parameters())
            self.model_ema.copy_to(self.unet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.use_ema:
                self.model_ema.restore(self.unet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.config.use_ema:
            self.model_ema(self.unet)

    def _pyramid_noise(self, epsilon, size_range=None, levels=10, scale_mode="nearest"):
        epsilon = epsilon.clone()
        multipliers = [1]
        for i in range(1, levels):
            m = 0.75**i
            h, w = epsilon.shape[-2] // 2**i, epsilon.shape[-2] // 2**i
            if size_range is None or (size_range[0] <= h <= size_range[1] or size_range[0] <= w <= size_range[1]):
                offset = paddle.randn(shape=[epsilon.shape[0], epsilon.shape[1], h, w])
                epsilon = (
                    epsilon + paddle.nn.functional.interpolate(x=offset, size=epsilon.shape[-2:], mode=scale_mode) * m
                )
                multipliers.append(m)
            if h <= 1 or w <= 1:
                break
        epsilon = epsilon / sum([(m**2) for m in multipliers]) ** 0.5
        return epsilon

    def forward(self, images: List[PIL.Image.Image] = None, captions: List[str] = None, **kwargs):
        batch = {
            "captions": captions,
            "images": images,
        }
        models = self.models
        extras = self.extras

        with paddle.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            epsilon = paddle.randn(shape=latents.shape, dtype=latents.dtype)
            epsilon = self._pyramid_noise(epsilon, size_range=[1, 16])
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(
                latents, shift=1, loss_shift=1, epsilon=epsilon
            )

        if self.config.dtype in ["float16", "bfloat16"]:
            with paddle.amp.auto_cast(dtype=self.config.dtype):
                pred = models.unet(noised, noise_cond, **conditions)
        else:
            pred = models.unet(noised, noise_cond, **conditions)

        loss = paddle.nn.functional.mse_loss(input=pred, label=target, reduction="none").mean(axis=[1, 2, 3])
        loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)

        # print("#####", "logSNR", logSNR.mean().item(), "loss_weight", loss_weight.mean().item(), "loss", loss.mean().item(), "noise_sum", noise.sum().item())

        return loss, loss_adjusted

    def models_to_save(self):
        return ["unet"]

    def encode_latents(self, batch: dict, models: Models, extras: Extras) -> paddle.Tensor:
        images = batch.get("images", None)
        return models.stage_a.encode(images)[0]

    @paddle.no_grad()
    def decode_latents(self, latents: paddle.Tensor, batch: dict, models: Models, extras: Extras) -> paddle.Tensor:
        return models.stage_a.decode(latents.astype(dtype="float32")).clip(min=0, max=1)

    @paddle.no_grad()
    def log_image(
        self,
        latent: paddle.Tensor,
        caption: str,
        height: int = 1024,
        width: int = 1024,
        batch_size: int = 1,
        seed: int = 1,
        steps: int = 20,
    ):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        extras_b = self.extras
        # Stage B Parameters
        extras_b.sampling_configs["cfg"] = 1.1
        extras_b.sampling_configs["shift"] = 1
        extras_b.sampling_configs["timesteps"] = 10
        extras_b.sampling_configs["t_start"] = 1.0

        # PREPARE CONDITIONS
        batch = {"captions": [caption] * batch_size}
        models_b = self.models
        conditions_b = self.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = self.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        with paddle.no_grad():
            with paddle.amp.auto_cast("float16"):
                paddle.seed(seed)

                conditions_b["effnet"] = latent
                unconditions_b["effnet"] = paddle.zeros_like(latent)

                sampling_b = extras_b.gdf.sample(
                    models_b.unet, conditions_b, stage_b_latent_shape, unconditions_b, **extras_b.sampling_configs
                )
                for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs["timesteps"]):
                    sampled_b = sampled_b
                image = models_b.stage_a.decode(sampled_b).cast("float32")
                image = image.clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.numpy().round().astype(np.uint8), sampled_b

    def set_ema(self, use_ema=False):
        self.config.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.unet)
