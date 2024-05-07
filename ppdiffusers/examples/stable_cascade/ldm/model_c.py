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
from ppdiffusers.models.stable_cascade.modules.previewer import Previewer
from ppdiffusers.models.stable_cascade.modules.stage_c import StageC
from ppdiffusers.transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

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


class ModelC(nn.Layer):
    class Config:
        dtype: str = config.dtype
        training: bool = False
        use_ema: bool = False
        model_version: str = config.model_c_version
        clip_image_model_name: str = config.clip_image_model_name
        clip_text_model_name: str = config.clip_text_model_name
        effnet_checkpoint_path: str = config.effnet_checkpoint_path
        previewer_checkpoint_path: str = config.previewer_checkpoint_path
        unet_checkpoint_path: str = config.stage_c_checkpoint_path
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
        previewer: paddle.nn.Layer = EXPECTED
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
            "cfg": 5,
            "sampler": DDPMSampler(gdf),
            "shift": 1,
            "timesteps": 20,
        }

        effnet_preprocess = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        clip_preprocess = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.Resize(224, interpolation="bicubic"),
                paddle.vision.transforms.CenterCrop(224),
                paddle.vision.transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess,
        )

    def load_models(self):
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
        previewer = Previewer()
        previewer_checkpoint = load_or_fail(self.config.previewer_checkpoint_path)
        previewer.set_state_dict(
            state_dict=previewer_checkpoint
            if "state_dict" not in previewer_checkpoint
            else previewer_checkpoint["state_dict"]
        )
        previewer.stop_gradient = True
        previewer.eval()
        previewer.to(dtype=dtype)
        del previewer_checkpoint

        # load unet
        if self.config.model_version == "3.6B":
            self.unet = StageC()
        elif self.config.model_version == "1B":
            self.unet = StageC(
                c_cond=1536,
                c_hidden=[1536, 1536],
                nhead=[24, 24],
                blocks=[[4, 12], [12, 4]],
            )
        else:
            raise ValueError(f"Unknown model version {self.config.model_version}")
        if self.config.unet_checkpoint_path is not None:
            self.unet.set_state_dict(load_or_fail(self.config.unet_checkpoint_path))
        self.unet.to(dtype=dtype)

        # load tokenizer, text_model, image_model
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(self.config.clip_text_model_name)
        text_model.eval()
        # text_model.to(dtype=self.config.dtype)

        image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name)
        image_model.eval()
        # image_model.to(dtype=self.config.dtype)

        # set models
        self.models = self.Models(
            tokenizer=tokenizer,
            text_model=text_model,
            image_model=image_model,
            previewer=previewer,
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
        images = paddle.to_tensor(images) if images is not None else None
        batch_size = len(captions)
        text_embeddings = None
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
            if "clip_text" in return_fields:
                text_embeddings = text_encoder_output.hidden_states[-1]
            if "clip_text_pooled" in return_fields:
                text_pooled_embeddings = text_encoder_output.text_embeds.unsqueeze(1)
        image_embeddings = None
        if "clip_img" in return_fields:
            image_embeddings = paddle.zeros(shape=[batch_size, 768])  # 全零的图像特征
        #     if images is not None:
        #         if is_eval:
        #             # 在测试阶段，直接使用真实图像特征
        #             if not is_unconditional and eval_image_embeds:
        #                 image_embeddings = models.image_model(
        #                     extras.clip_preprocess(images)
        #                 ).image_embeds
        #         else:
        #             pass
        #             # 在训练阶段，随机采样部分图像特征，其余全零
        #             rand_idx = np.random.rand(batch_size) > 0.9
        #             if np.any(rand_idx):
        #                 for i, img in enumerate(images):
        #                     if rand_idx[i]:
        #                         image_embeddings[i] = models.image_model(
        #                             extras.clip_preprocess(img).unsqueeze(0)
        #                         ).image_embeds[0]
        #     image_embeddings = image_embeddings.unsqueeze(axis=1)
        # text_embeddings = paddle.to_tensor(np.load("/root/lxl/0_SC/Paddle-SC/text_embeddings_2.npy"))
        # text_pooled_embeddings = paddle.to_tensor(np.load("/root/lxl/0_SC/Paddle-SC/text_pooled_embeddings_2.npy"))
        # image_embeddings = paddle.to_tensor(np.load("/root/lxl/0_SC/Paddle-SC/image_embeddings_2.npy"))

        return {
            # "clip_text": paddle.to_tensor(text_embeddings.cpu().detach().numpy()),
            # "clip_text_pooled": paddle.to_tensor(text_pooled_embeddings.cpu().detach().numpy()),
            # # "clip_img": image_embeddings.cast(self.config.dtype),
            # "clip_img": paddle.to_tensor(image_embeddings.cpu().detach().numpy()),
            "clip_text": text_embeddings,
            "clip_text_pooled": text_pooled_embeddings,
            "clip_img": image_embeddings,
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
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)

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
        image_tensor = extras.effnet_preprocess(images)
        image_tensor = image_tensor.cast("float32")

        models.effnet.eval()
        latent = models.effnet(image_tensor).cast(self.config.dtype)

        return latent

    @paddle.no_grad()
    def decode_latents(self, latents: paddle.Tensor, batch: dict, models: Models, extras: Extras) -> paddle.Tensor:
        latents = latents.cast("float32")
        return models.previewer(latents)

    @paddle.no_grad()
    def log_image(
        self,
        caption: str = "a book",
        height: int = 1024,
        width: int = 1024,
        cfg: int = 4,
        batch_size: int = 1,
        seed: int = 1,
        steps: int = 20,
    ):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        # Stage C Parameters
        # extras = self.setup_extras_pre()
        extras = self.extras
        extras.sampling_configs["cfg"] = cfg
        extras.sampling_configs["shift"] = 2
        extras.sampling_configs["timesteps"] = steps
        extras.sampling_configs["t_start"] = 1.0

        # PREPARE CONDITIONS
        batch = {"captions": [caption] * batch_size}
        models = self.models
        conditions = self.get_conditions(
            batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False
        )
        unconditions = self.get_conditions(
            batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False
        )

        with paddle.no_grad():
            with paddle.amp.auto_cast("float16"):
                paddle.seed(seed)

                sampling_c = extras.gdf.sample(
                    models.unet,
                    conditions,
                    stage_c_latent_shape,
                    unconditions,
                    **extras.sampling_configs,
                )
                for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs["timesteps"]):
                    sampled_c = sampled_c

                sampled_c = sampled_c.cast("float32")
                image = models.previewer(sampled_c)
                image = image.clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.numpy().round().astype(np.uint8), sampled_c

    def set_ema(self, use_ema=False):
        self.config.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.unet)
