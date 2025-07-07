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
import paddle

from ppdiffusers import FluxPipeline

from .utils.schedulers import get_scheduler


class PaddleFLUXCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"),)}}

    RETURN_TYPES = ("PIPELINE",)
    RETURN_NAMES = ("flux_pipe",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "🚢 paddlemix/ppdiffusers/input"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        pipe = FluxPipeline.from_single_file(ckpt_path)
        return (pipe,)


class PaddleFLUXText2ImagePipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_pipe": ("PIPELINE",),
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
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "number": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 99999999999999999999999}),
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
                        "flowmatch-euler",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "🚢 paddlemix/ppdiffusers/pipelines"

    def sample(self, flux_pipe, prompt, negative_prompt, steps, width, height, number, seed, cfg, scheduler_type):

        pipe = FluxPipeline(**flux_pipe.components)
        pipe.scheduler = get_scheduler(scheduler_type)
        paddle.seed(seed)

        # progress_bar = ProgressBar(steps)
        latent = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=number,
            num_inference_steps=steps,
            guidance_scale=cfg,
            output_type="pil",
        ).images

        return (latent,)


NODE_CLASS_MAPPINGS = {
    "PaddleFLUXCheckpointLoader": PaddleFLUXCheckpointLoader,
    "PaddleFLUXText2ImagePipe": PaddleFLUXText2ImagePipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleFLUXCheckpointLoader": "Paddle FLUX Checkpoint Loader",
    "PaddleFLUXText2ImagePipe": "Paddle FLUX Text2Image Pipe",
}
