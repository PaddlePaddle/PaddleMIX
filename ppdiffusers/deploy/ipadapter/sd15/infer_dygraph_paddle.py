# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import time
import warnings

import cv2
import numpy as np
import paddle
from PIL import Image
from tqdm.auto import trange

from ppdiffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from ppdiffusers.utils import load_image


def get_canny_image(image, args):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.Canny(image, args.low_threshold, args.high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def change_scheduler(self, scheduler_type="ddim"):
    self.original_scheduler_config = self.scheduler.config
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler.from_config(self.original_scheduler_config, skip_prk_steps=True)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "dpm-multi":
        scheduler = DPMSolverMultistepScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "dpm-single":
        scheduler = DPMSolverSinglestepScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "kdpm2-ancestral":
        scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "kdpm2":
        scheduler = KDPM2DiscreteScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "unipc-multi":
        scheduler = UniPCMultistepScheduler.from_config(self.original_scheduler_config)
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(
            self.original_scheduler_config,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
        )
    elif scheduler_type == "ddpm":
        scheduler = DDPMScheduler.from_config(
            self.original_scheduler_config,
        )
    elif scheduler_type == "deis-multi":
        scheduler = DEISMultistepScheduler.from_config(
            self.original_scheduler_config,
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    return scheduler


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the bos).",
    )
    parser.add_argument(
        "--ipadapter_pretrained_model_name_or_path",
        type=str,
        default="h94/IP-Adapter",
        help="Path to the `ppdiffusers`'s ip-adapter checkpoint to convert (either a local directory or on the bos).Example: h94/IP-Adapter",
    )
    parser.add_argument(
        "--ipadapter_model_subfolder",
        type=str,
        default="models",
        help="Path to the `ppdiffusers`'s ip-adapter checkpoint to convert (either a local directory or on the bos).Example: models",
    )
    parser.add_argument(
        "--ipadapter_weight_name",
        type=str,
        default="ip-adapter_sd15.safetensors",
        help="Name of the weight to convert.Example: ip-adapter_sd15.safetensors",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="The number of unet inference steps.",
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=10,
        help="The number of performance benchmark steps.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="all",
        choices=[
            "text2img",
            "img2img",
            "inpaint_legacy",
            "all",
        ],
        help="The task can be one of [text2img, img2img, inpaint_legacy, all]. ",
    )
    parser.add_argument(
        "--parse_prompt_type",
        type=str,
        default="raw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Whether to use FP16 mode")
    parser.add_argument(
        "--attention_type", type=str, default="raw", choices=["raw", "cutlass", "flash", "all"], help="attention_type."
    )
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ],
        help="The scheduler type of stable diffusion.",
    )
    parser.add_argument("--height", type=int, default=512, help="Height of input image")
    parser.add_argument("--width", type=int, default=512, help="Width of input image")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength for img2img / inpaint")
    return parser.parse_args()


def main(args):

    seed = 1024
    paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        paddle_dtype=paddle_dtype,
    )
    pipe.load_ip_adapter(
        args.ipadapter_pretrained_model_name_or_path,
        subfolder=args.ipadapter_model_subfolder,
        weight_name=args.ipadapter_weight_name,
    )
    scheduler = change_scheduler(pipe, args.scheduler)
    pipe.scheduler = scheduler

    if args.attention_type == "all":
        args.attention_type = ["raw", "cutlass", "flash"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        if attention_type == "raw":
            pipe.disable_xformers_memory_efficient_attention()
        else:
            try:
                pipe.enable_xformers_memory_efficient_attention(attention_type)
            except Exception as e:
                if attention_type == "flash":
                    warnings.warn(
                        "Attention type flash is not supported on your GPU! We need to use 3060、3070、3080、3090、4060、4070、4080、4090、A30、A100 etc."
                    )
                    continue
                else:
                    raise ValueError(e)

        if not args.use_fp16 and attention_type == "flash":
            print("Flash attention is not supported dtype=float32! Please use float16 or bfloat16. We will skip this!")
            continue

        width = args.width
        height = args.height
        pipe.set_progress_bar_config(disable=False)

        img_url = "https://paddlenlp.bj.bcebos.com/models/community/examples/images/load_neg_embed.png"
        ip_image = load_image(img_url)

        folder = f"paddle_attn_{attention_type}_fp16" if args.use_fp16 else f"paddle_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)
        if args.task_name in ["text2img", "all"]:
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            # text2img
            prompt = "a photo of an astronaut riding a horse on mars"
            time_costs = []
            # warmup
            pipe(
                prompt,
                num_inference_steps=10,
                height=height,
                width=width,
                ip_adapter_image=ip_image,
            )
            print("==> Test text2img performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe(
                    prompt,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    ip_adapter_image=ip_image,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Attention type: {attention_type}, "
                f"Use fp16: {'true' if args.use_fp16 else 'false'}, "
                f"Mean iter/sec: {1 / (np.mean(time_costs) / args.inference_steps):2f} it/s, "
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/text2img.png")

        if args.task_name in ["img2img", "all"]:
            pipe_img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
            pipe_img2img.set_progress_bar_config(disable=False)
            img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
            init_image = load_image(img_url).resize((width, height))
            prompt = "A fantasy landscape, trending on artstation"
            time_costs = []
            # warmup
            pipe_img2img(
                prompt,
                image=init_image,
                num_inference_steps=20,
                height=height,
                width=width,
                strength=args.strength,
                ip_adapter_image=ip_image,
            )
            print("==> Test img2img performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe_img2img(
                    prompt,
                    image=init_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    strength=args.strength,
                    ip_adapter_image=ip_image,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Attention type: {attention_type}, "
                f"Use fp16: {'true' if args.use_fp16 else 'false'}, "
                f"Mean iter/sec: {1 / (np.mean(time_costs) / args.inference_steps):2f} it/s, "
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/img2img.png")

        if args.task_name in ["inpaint_legacy", "all"]:
            pipe_inpaint = StableDiffusionInpaintPipeline(**pipe.components)
            pipe_inpaint.set_progress_bar_config(disable=False)
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
            )
            mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
            init_image = load_image(img_url).resize((width, height))
            mask_image = load_image(mask_url).resize((width, height))
            prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            time_costs = []
            task_name = "inpaint_legacy"
            pipe_inpaint(
                prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=20,
                height=height,
                width=width,
                strength=args.strength,
                ip_adapter_image=ip_image,
            )
            print(f"==> Test {task_name} performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe_inpaint(
                    prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    strength=args.strength,
                    ip_adapter_image=ip_image,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Attention type: {attention_type}, "
                f"Use fp16: {'true' if args.use_fp16 else 'false'}, "
                f"Mean iter/sec: {1 / (np.mean(time_costs) / args.inference_steps):2f} it/s, "
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/{task_name}.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
