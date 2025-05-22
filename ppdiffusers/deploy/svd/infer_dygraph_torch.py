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

import numpy as np
import torch
from diffusers import (
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
    StableVideoDiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from tqdm.auto import trange


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
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the bos).",
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
            "img2video",
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
    # parser.add_argument(
    #     "--attention_type", type=str, default="raw", choices=["raw", "cutlass", "flash", "all"], help="attention_type."
    # )
    # currently, torch3 not support flash attention
    parser.add_argument(
        "--attention_type", type=str, default="raw", choices=["raw", "cutlass", "all"], help="attention_type."
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
    parser.add_argument("--height", type=int, default=576, help="The height of output images. Default: None")
    parser.add_argument("--width", type=int, default=1024, help="The width of output images. Default: None")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="The number of video frames to generate. Defaults: None, \
        resulting to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`",
    )
    return parser.parse_args()


def main(args):

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
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

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)
        if args.task_name in ["img2video", "all"]:
            # img2video
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/rocket.png"
            )
            init_image = load_image(img_url)
            time_costs = []
            # warmup
            print("==> Warmup.")
            pipe(
                image=init_image,
                num_inference_steps=3,
                height=height,
                width=width,
                fps=7,
                decode_chunk_size=2,
            )
            print("==> Test img2video performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                torch.cuda.manual_seed(seed)
                frames = pipe(
                    image=init_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    fps=7,
                    decode_chunk_size=2,
                ).frames
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
            frames[0][0].save(f"{folder}/test_svd.gif", save_all=True, append_images=frames[0][1:], loop=0)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
