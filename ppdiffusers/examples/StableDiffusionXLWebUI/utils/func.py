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

import gc
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import paddle
from PIL import PngImagePlugin

from ppdiffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)

MAX_SEED = np.iinfo(np.int32).max


def similar_img_generate(metadata, image):
    from .webui import img2img, inp2img, text2img

    if metadata["type_to_img"] == "text":
        return text2img(
            pipe=metadata.get("pipe"),
            prompt=metadata.get("prompt", "masterpiece"),
            negative_prompt=metadata.get("negative_prompt", ""),
            guidance_scale=int(metadata.get("guidance_scale", 4)),
            height=int(metadata.get("height", 1024)),
            width=int(metadata.get("width", 1024)),
            num_inference_steps=int(metadata.get("num_inference_steps", 30)),
            seed=int(metadata.get("seed", -1)),
            scheduler_type=metadata.get("scheduler_type", "default"),
            num_images_per_prompt=1,
            model_name_input=None,
            enable_xformers="ON",
            vae_dir="",
            vae_dir_input="",
            max_size=1900,
            rolers="KeQing",
            load_to_cpu="OFF",
            type_to_img="text",
        )
    elif metadata["type_to_img"] == "img":
        return img2img(
            pipe=metadata.get("pipe"),
            prompt=metadata.get("prompt", "masterpiece"),
            img_image=image,
            negative_prompt=metadata.get("negative_prompt", ""),
            guidance_scale=int(metadata.get("guidance_scale", 4)),
            height=int(metadata.get("height", 1024)),
            width=int(metadata.get("width", 1024)),
            num_inference_steps=int(metadata.get("num_inference_steps", 30)),
            strength=float(metadata.get("strength", 0.5)),
            seed=int(metadata.get("seed", -1)),
            scheduler_type=metadata.get("scheduler_type", "default"),
            num_images_per_prompt=1,
            model_name_input=None,
            enable_xformers="ON",
            vae_dir="",
            vae_dir_input="",
            max_size=1800,
            rolers="KeQing",
            load_to_cpu="OFF",
            type_to_img="img",
        )
    elif metadata["type_to_img"] == "inp":
        return inp2img(
            pipe=metadata.get("pipe"),
            prompt=metadata.get("prompt", "masterpiece"),
            img_image=image,
            negative_prompt=metadata.get("negative_prompt", ""),
            guidance_scale=int(metadata.get("guidance_scale", 4)),
            height=int(metadata.get("height", 1024)),
            width=int(metadata.get("width", 1024)),
            num_inference_steps=int(metadata.get("num_inference_steps", 30)),
            strength=float(metadata.get("strength", 0.8)),
            scheduler_type=metadata.get("scheduler_type", "default"),
            seed=int(metadata.get("seed", -1)),
            num_images_per_prompt=1,
            model_name_input=None,
            enable_xformers="ON",
            vae_dir="",
            vae_dir_input="",
            max_size=1800,
            rolers="KeQing",
            load_to_cpu="OFF",
            type_to_img="img",
        )


@dataclass
class StyleConfig:
    prompt: str
    negative_prompt: str


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def seed_everything(seed: int) -> paddle.Generator:
    np.random.seed(seed)
    generator = paddle.Generator()
    generator.manual_seed(seed)
    return generator


def parse_aspect_ratio(aspect_ratio: str) -> Optional[Tuple[int, int]]:
    if aspect_ratio == "Custom":
        return None
    width, height = aspect_ratio.split(" x ")
    return int(width), int(height)


def aspect_ratio_handler(aspect_ratio: str, custom_width: int, custom_height: int) -> Tuple[int, int]:
    if aspect_ratio == "Custom":
        return custom_width, custom_height
    else:
        width, height = parse_aspect_ratio(aspect_ratio)
        return width, height


def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        ),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(scheduler_config),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()


def free_memory() -> None:
    paddle.device.cuda.empty_cache()
    gc.collect()


def preprocess_prompt(
    style_dict,
    style_name: str,
    positive: str,
    negative: str = "",
    add_style: bool = True,
) -> Tuple[str, str]:
    p, n = style_dict.get(style_name, style_dict["无"])

    if add_style and positive.strip():
        formatted_positive = p.format(prompt=positive)
    else:
        formatted_positive = positive

    combined_negative = n
    if negative.strip():
        if combined_negative:
            combined_negative += ", " + negative
        else:
            combined_negative = negative

    return formatted_positive, combined_negative


def common_upscale(
    samples: paddle.Tensor,
    width: int,
    height: int,
    upscale_method: str,
) -> paddle.Tensor:
    return paddle.nn.functional.interpolate(samples, size=(height, width), mode=upscale_method)


def upscale(samples: paddle.Tensor, upscale_method: str, scale_by: float) -> paddle.Tensor:
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    return common_upscale(samples, width, height, upscale_method)


def load_wildcard_files(wildcard_dir: str) -> Dict[str, str]:
    wildcard_files = {}
    for file in os.listdir(wildcard_dir):
        if file.endswith(".txt"):
            key = f"__{file.split('.')[0]}__"  # Create a key like __character__
            wildcard_files[key] = os.path.join(wildcard_dir, file)
    return wildcard_files


def get_random_line_from_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        lines = file.readlines()
        if not lines:
            return ""
        return random.choice(lines).strip()


def add_wildcard(prompt: str, wildcard_files: Dict[str, str]) -> str:
    for key, file_path in wildcard_files.items():
        if key in prompt:
            wildcard_line = get_random_line_from_file(file_path)
            prompt = prompt.replace(key, wildcard_line)
    return prompt


def preprocess_image_dimensions(width, height):
    if width % 8 != 0:
        width = width - (width % 8)
    if height % 8 != 0:
        height = height - (height % 8)
    return width, height


def save_image(image, metadata, output_dir):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"image_{current_time}.png"
    filepath = os.path.join(output_dir, filename)

    metadata_str = json.dumps(metadata)
    info = PngImagePlugin.PngInfo()
    info.add_text("metadata", metadata_str)
    image.save(filepath, "PNG", pnginfo=info)
    return filepath
