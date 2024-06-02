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

from typing import List

import paddle
from PIL import Image

from ppdiffusers.pipelines.controlnet import MultiControlNetModel
from ppdiffusers.transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from ppdiffusers.utils import smart_load

from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler
from .utils import swich_state


class ImageProjModel(paddle.nn.Layer):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = paddle.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = paddle.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            [-1, self.clip_extra_context_tokens, self.cross_attention_dim]
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(paddle.nn.Layer):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = paddle.nn.Sequential(
            paddle.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            paddle.nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        if isinstance(image_encoder_path, str):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.image_encoder_path,
                paddle_dtype=paddle.float16,
            ).to(self.device)
        else:
            self.image_encoder = image_encoder_path
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(self.device, dtype=paddle.float16)
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=paddle.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if ".pd" in self.ip_ckpt:
            f = paddle.load(self.ip_ckpt)
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            for key in list(f.keys()):
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.pop(key).cast(paddle.float32)
                if key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.pop(key).cast(paddle.float16)
        if ".safetensors" in self.ip_ckpt:
            f = smart_load(self.ip_ckpt)
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            for key in list(f.keys()):
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.pop(key).cast(paddle.float32)
                if key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.pop(key).cast(paddle.float16)
            state_dict["image_proj"] = swich_state(state_dict["image_proj"], dtype="float32")
            state_dict["ip_adapter"] = swich_state(state_dict["ip_adapter"], dtype="float16")
        if ".bin" in self.ip_ckpt:
            state_dict = smart_load(self.ip_ckpt)
            state_dict["image_proj"] = swich_state(state_dict["image_proj"], dtype="float32")
            state_dict["ip_adapter"] = swich_state(state_dict["ip_adapter"], dtype="float16")
        if isinstance(self.ip_ckpt, dict):
            state_dict = self.ip_ckpt

        self.image_proj_model.set_state_dict(state_dict["image_proj"])
        ip_layers = paddle.nn.LayerList(self.pipe.unet.attn_processors.values())  # .to(dtype=paddle.float16)
        ip_layers.set_state_dict(state_dict["ip_adapter"])

    @paddle.no_grad()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pd").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds.astype("float32"))
        uncond_image_prompt_embeds = self.image_proj_model(paddle.zeros_like(clip_image_embeds.astype("float32")))
        return image_prompt_embeds.astype("float16"), uncond_image_prompt_embeds.astype("float16")

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.tile([1, num_samples, 1])
        image_prompt_embeds = image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.tile([1, num_samples, 1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])

        with paddle.no_grad():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(  # xl差异
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = paddle.concat([prompt_embeds_, image_prompt_embeds], axis=1)
            negative_prompt_embeds = paddle.concat([negative_prompt_embeds_, uncond_image_prompt_embeds], axis=1)
        if kwargs.get("generator", None) is not None:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                **kwargs,
            )
            return images

        generator = paddle.Generator().manual_seed(seed) if seed is not None else None
        images = self.pipe(  # xl差异
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )
        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.tile([1, num_samples, 1])
        image_prompt_embeds = image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.tile([1, num_samples, 1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])

        with paddle.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = paddle.concat([prompt_embeds, image_prompt_embeds], axis=1)
            negative_prompt_embeds = paddle.concat([negative_prompt_embeds, uncond_image_prompt_embeds], axis=1)
        if kwargs.get("generator", None) is not None:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                **kwargs,
            )
            return images

        generator = paddle.Generator().manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=paddle.float16)
        return image_proj_model

    @paddle.no_grad()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pd").pixel_values
        clip_image = clip_image.to(self.device).to(dtype=paddle.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            paddle.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device)
        return image_proj_model

    @paddle.no_grad()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pd").pixel_values
        clip_image = clip_image.to(self.device).to(dtype=paddle.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds.astype("float32"))
        uncond_clip_image_embeds = self.image_encoder(
            paddle.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds.astype("float32"))
        return image_prompt_embeds.astype("float16"), uncond_image_prompt_embeds.astype("float16")

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.tile([1, num_samples, 1])
        image_prompt_embeds = image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.tile([1, num_samples, 1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])

        with paddle.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        prompt_embeds = paddle.concat([prompt_embeds, image_prompt_embeds], axis=1)
        negative_prompt_embeds = paddle.concat([negative_prompt_embeds, uncond_image_prompt_embeds], axis=1)
        if kwargs.get("generator", None) is not None:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                **kwargs,
            )
            return images

        generator = paddle.Generator().manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images
