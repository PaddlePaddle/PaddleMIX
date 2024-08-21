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


import folder_paths
import numpy as np
import paddle
import torch  # for convert data
from comfy.utils import ProgressBar

from ppdiffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from .utils.schedulers import get_scheduler


class PaddleSDXLCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"),)}}

    RETURN_TYPES = ("PIPELINE",)
    RETURN_NAMES = ("sd_pipe",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "🚢 paddlemix/ppdiffusers/input"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path)
        return (pipe,)


class PaddleSDXLVaeDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "sd_pipe": ("PIPELINE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "🚢 paddlemix/ppdiffusers/output"

    def decode(self, sd_pipe, latent):
        vae = sd_pipe.vae
        latent = 1 / vae.config.scaling_factor * latent
        image = vae.decode(latent, return_dict=False)[0]
        image = (image / 2 + 0.5).clip(0, 1)
        image = image.cast(dtype=paddle.float32).transpose([0, 2, 3, 1]).cpu().numpy()
        image = (image * 255).astype(np.uint8)

        return (image,)


class PaddleSDXLText2ImagePipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_pipe": ("PIPELINE",),
                "prompt": ("PROMPT",),
                "negative_prompt": ("PROMPT",),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "number": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999999999999999999}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                    },
                ),
                "scheduler_type": (
                    [
                        "euler",
                        "euler-ancestral",
                        "pndm",
                        "lms",
                        "heun",
                        "dpm-multi",
                        "dpm-single",
                        "kdpm2-ancestral",
                        "kdpm2",
                        "unipc-multi",
                        "ddim",
                        "ddpm",
                        "deis-multi",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "🚢 paddlemix/ppdiffusers/pipelines"

    def sample(self, sd_pipe, prompt, negative_prompt, steps, width, height, number, seed, cfg, scheduler_type):

        pipe = StableDiffusionXLPipeline(**sd_pipe.components)
        pipe.scheduler = get_scheduler(scheduler_type)
        paddle.seed(seed)

        progress_bar = ProgressBar(steps)
        latent = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=number,
            num_inference_steps=steps,
            guidance_scale=cfg,
            output_type="latent",
            callback=lambda step, timestep, latents: progress_bar.update_absolute(
                value=step, total=steps, preview=None
            ),
            callback_steps=1,
        ).images

        return (latent,)


class PaddleSDXLImage2ImagePipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_pipe": ("PIPELINE",),
                "image": ("IMAGE",),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "prompt": ("PROMPT",),
                "negative_prompt": ("PROMPT",),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "number": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999999999999999999}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                    },
                ),
                "scheduler_type": (
                    [
                        "euler",
                        "euler-ancestral",
                        "pndm",
                        "lms",
                        "heun",
                        "dpm-multi",
                        "dpm-single",
                        "kdpm2-ancestral",
                        "kdpm2",
                        "unipc-multi",
                        "ddim",
                        "ddpm",
                        "deis-multi",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "🚢 paddlemix/ppdiffusers/pipelines"

    def sample(self, sd_pipe, image, denoise, prompt, negative_prompt, steps, number, seed, cfg, scheduler_type):
        # torch -> numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        pipe = StableDiffusionXLImg2ImgPipeline(**sd_pipe.components)
        pipe.scheduler = get_scheduler(scheduler_type)
        paddle.seed(seed)

        progress_bar = ProgressBar(steps)
        latent = pipe(
            image=image,
            strength=denoise,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=number,
            num_inference_steps=steps,
            guidance_scale=cfg,
            output_type="latent",
            callback=lambda step, timestep, latents: progress_bar.update_absolute(
                value=step, total=steps, preview=None
            ),
            callback_steps=1,
        ).images

        return (latent,)


class PaddleSDXLInpaintPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_pipe": ("PIPELINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "prompt": ("PROMPT",),
                "negative_prompt": ("PROMPT",),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "number": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999999999999999999}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                    },
                ),
                "scheduler_type": (
                    [
                        "euler",
                        "euler-ancestral",
                        "pndm",
                        "lms",
                        "heun",
                        "dpm-multi",
                        "dpm-single",
                        "kdpm2-ancestral",
                        "kdpm2",
                        "unipc-multi",
                        "ddim",
                        "ddpm",
                        "deis-multi",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "🚢 paddlemix/ppdiffusers/pipelines"

    def sample(self, sd_pipe, image, mask, denoise, prompt, negative_prompt, steps, number, seed, cfg, scheduler_type):
        # torch -> numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        height, width = image.shape[1] // 8 * 8, image.shape[2] // 8 * 8

        pipe = StableDiffusionXLInpaintPipeline(**sd_pipe.components)
        pipe.scheduler = get_scheduler(scheduler_type)
        paddle.seed(seed)

        progress_bar = ProgressBar(steps)
        latent = pipe(
            image=image,
            mask_image=mask,
            strength=denoise,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=number,
            num_inference_steps=steps,
            guidance_scale=cfg,
            output_type="latent",
            callback=lambda step, timestep, latents: progress_bar.update_absolute(
                value=step, total=steps, preview=None
            ),
            callback_steps=1,
        ).images

        return (latent,)


NODE_CLASS_MAPPINGS = {
    "PaddleSDXLCheckpointLoader": PaddleSDXLCheckpointLoader,
    "PaddleSDXLVaeDecoder": PaddleSDXLVaeDecoder,
    "PaddleSDXLText2ImagePipe": PaddleSDXLText2ImagePipe,
    "PaddleSDXLImage2ImagePipe": PaddleSDXLImage2ImagePipe,
    "PaddleSDXLInpaintPipe": PaddleSDXLInpaintPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleSDXLCheckpointLoader": "Paddle SDXL Checkpoint Loader",
    "PaddleSDXLVaeDecoder": "Paddle SDXL VAE Decoder",
    "PaddleSDXLText2ImagePipe": "Paddle SDXL Text2Image Pipe",
    "PaddleSDXLImage2ImagePipe": "Paddle SDXL Image2Image Pipe",
    "PaddleSDXLInpaintPipe": "Paddle SDXL Inpaint Pipe",
}
