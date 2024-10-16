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

from ppdiffusers.transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from ppdiffusers.utils import smart_load

from .attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from .resampler import FeedForward, PerceiverAttention
from .utils import swich_state


class FacePerceiverResampler(paddle.nn.Layer):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = paddle.nn.Linear(embedding_dim, dim)
        self.proj_out = paddle.nn.Linear(dim, output_dim)
        self.norm_out = paddle.nn.LayerNorm(output_dim)
        self.layers = paddle.nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                paddle.nn.LayerList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class MLPProjModel(paddle.nn.Layer):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = paddle.nn.Sequential(
            paddle.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            paddle.nn.GELU(),
            paddle.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = paddle.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)  # [id_embeds.shape[0], num_tokens * cross_attention_dim]
        x = x.reshape(
            [-1, self.num_tokens, self.cross_attention_dim]
        )  # [id_embeds.shape[0], num_tokens, cross_attention_dim]
        x = self.norm(x)
        return x


class ProjPlusModel(paddle.nn.Layer):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = paddle.nn.Sequential(
            paddle.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            paddle.nn.GELU(),
            paddle.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = paddle.nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):

        x = self.proj(id_embeds)
        x = x.reshape([-1, self.num_tokens, self.cross_attention_dim])
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class IPAdapterFaceID:
    def __init__(self, sd_pipe, ip_ckpt, device, lora_rank=128, num_tokens=4, paddle_dtype=paddle.float16):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.lora_rank = lora_rank
        self.num_tokens = num_tokens
        self.paddle_dtype = paddle_dtype
        self.pipe = sd_pipe
        self.set_ip_adapter()

        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens,
        )
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
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                ).to(dtype=unet.dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    rank=self.lora_rank,
                    num_tokens=self.num_tokens,
                ).to(dtype=unet.dtype)
        unet.set_attn_processor(attn_procs)

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

        self.image_proj_model.set_state_dict(state_dict["image_proj"])
        ip_layers = paddle.nn.LayerList(self.pipe.unet.attn_processors.values())
        ip_layers.set_state_dict(state_dict["ip_adapter"])

    @paddle.no_grad()
    def get_image_embeds(self, faceid_embeds):
        multi_face = False
        if len(faceid_embeds.shape) == 3:
            multi_face = True
            b, n, c = faceid_embeds.shape
            faceid_embeds = faceid_embeds.reshape([b * n, c])
        faceid_embeds = faceid_embeds.cast(paddle.float32)
        image_prompt_embeds = self.image_proj_model(faceid_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(paddle.zeros_like(faceid_embeds))
        if multi_face:
            c = image_prompt_embeds.shape[-1]
            image_prompt_embeds = image_prompt_embeds.reshape([b, -1, c])
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape([b, -1, c])
        return image_prompt_embeds.cast(self.paddle_dtype), uncond_image_prompt_embeds.cast(self.paddle_dtype)

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)
        self.set_scale(scale)

        num_prompts = faceid_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.tile([1, num_samples, 1])
        image_prompt_embeds = image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.tile([1, num_samples, 1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])

        with paddle.no_grad():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
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
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images


class IPAdapterFaceIDPlus:
    def __init__(
        self, sd_pipe, image_encoder_path, ip_ckpt, device, lora_rank=128, num_tokens=4, paddle_dtype=paddle.float16
    ):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.lora_rank = lora_rank
        self.num_tokens = num_tokens
        self.paddle_dtype = paddle_dtype

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_path, paddle_dtype=self.paddle_dtype
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ProjPlusModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens,
        )
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
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                )
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    rank=self.lora_rank,
                    num_tokens=self.num_tokens,
                )
        unet.set_attn_processor(attn_procs)

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

        ip_layers = paddle.nn.LayerList(self.pipe.unet.attn_processors.values()).to(dtype=paddle.float16)
        ip_layers.set_state_dict(state_dict["ip_adapter"])

    @paddle.no_grad()
    def get_image_embeds(self, faceid_embeds, face_image, s_scale, shortcut):
        if isinstance(face_image, Image.Image):
            face_image = [face_image]
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pd").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.paddle_dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(
            paddle.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]

        faceid_embeds = faceid_embeds.cast(paddle.float32)
        image_prompt_embeds = self.image_proj_model(
            faceid_embeds, clip_image_embeds.cast(paddle.float32), shortcut=shortcut, scale=s_scale
        )
        uncond_image_prompt_embeds = self.image_proj_model(
            paddle.zeros_like(faceid_embeds),
            uncond_clip_image_embeds.cast(paddle.float32),
            shortcut=shortcut,
            scale=s_scale,
        )
        return image_prompt_embeds.cast(self.paddle_dtype), uncond_image_prompt_embeds.cast(self.paddle_dtype)

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        face_image=None,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        s_scale=1.0,
        shortcut=False,
        **kwargs,
    ):
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)
        self.set_scale(scale)

        num_prompts = faceid_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            faceid_embeds, face_image, s_scale, shortcut
        )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.tile([1, num_samples, 1])
        image_prompt_embeds = image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.tile([1, num_samples, 1])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape([bs_embed * num_samples, seq_len, -1])

        with paddle.no_grad():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
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
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images


class IPAdapterFaceIDXL(IPAdapterFaceID):
    """SDXL"""

    def generate(
        self,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)
        self.set_scale(scale)

        num_prompts = faceid_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds)

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
        paddle.device.cuda.empty_cache()

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
        paddle.device.cuda.empty_cache()
        return images


class IPAdapterFaceIDPlusXL(IPAdapterFaceIDPlus):
    """SDXL"""

    def generate(
        self,
        face_image=None,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        s_scale=1.0,
        shortcut=True,
        **kwargs,
    ):
        if kwargs.get("num_images_per_prompt", None) is not None:
            num_samples = kwargs.get("num_images_per_prompt", None)
        self.set_scale(scale)

        num_prompts = faceid_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            faceid_embeds, face_image, s_scale, shortcut
        )

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
            guidance_scale=guidance_scale,
            **kwargs,
        )
        self.pipe.to("cpu")
        paddle.device.cuda.empty_cache()

        return images
