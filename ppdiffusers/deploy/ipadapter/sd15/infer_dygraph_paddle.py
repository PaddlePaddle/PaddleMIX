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

# isort: split
import paddle

# isort: split
import numpy as np
from paddlenlp.trainer.argparser import strtobool
from tqdm.auto import trange

from ppdiffusers import (  # noqa
    DiffusionPipeline,
    StableDiffusionMegaPipeline,
    StableDiffusionPipeline,
)
from ppdiffusers.utils import load_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the `ppdiffusers` checkpoint to convert (either a local directory or on the bos).",
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
        default=1,
        help="The number of performance benchmark steps.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        choices=["paddle"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=[
            "cpu",
            "gpu",
            "huawei_ascend_npu",
            "kunlunxin_xpu",
        ],
        help="The inference runtime device of models.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="text2img",
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
        default="lpw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="preconfig-euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "preconfig-euler-ancestral",
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
    parser.add_argument("--hr_resize_height", type=int, default=768, help="HR Height of input image")
    parser.add_argument("--hr_resize_width", type=int, default=768, help="HR Width of input image")
    parser.add_argument("--is_sd2_0", type=strtobool, default=False, help="Is sd2_0 model?")
    parser.add_argument(
        "--tune",
        type=strtobool,
        default=False,
        help="Whether to tune the shape of tensorrt engine.",
    )

    return parser.parse_args()


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    seed = 1024
    min_image_size = 512
    max_image_size = 768
    max_image_size = max(min_image_size, max_image_size)
    pipe = StableDiffusionMegaPipeline.from_pretrained(
        args.model_dir,
    )
    pipe.load_ip_adapter(
        args.ipadapter_pretrained_model_name_or_path,
        subfolder=args.ipadapter_model_subfolder,
        weight_name=args.ipadapter_weight_name,
    )
    pipe.set_progress_bar_config(disable=False)
    pipe.change_scheduler(args.scheduler)
    parse_prompt_type = args.parse_prompt_type
    width = args.width
    height = args.height

    folder = f"results-{args.backend}"
    os.makedirs(folder, exist_ok=True)
    if args.task_name in ["text2img", "all"]:
        # text2img
        prompt = "a photo of an astronaut riding a horse on mars"
        time_costs = []
        # warmup
        img_url = "https://paddlenlp.bj.bcebos.com/models/community/examples/images/load_neg_embed.png"
        ip_image = load_image(img_url)
        pipe.text2img(
            prompt,
            num_inference_steps=20,
            height=height,
            width=width,
            ip_adapter_image=ip_image,
            parse_prompt_type=parse_prompt_type,
        )
        print("==> Test text2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            images = pipe.text2img(
                prompt,
                output_type="pil",
                num_inference_steps=args.inference_steps,
                height=height,
                width=width,
                ip_adapter_image=ip_image,
                parse_prompt_type=parse_prompt_type,
            ).images
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        images[0].save(f"{folder}/text2img.png")

    if args.task_name in ["img2img", "all"]:
        # img2img
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
        )
        init_image = load_image(img_url)
        ip_image = load_image(img_url)
        prompt = "A fantasy landscape, trending on artstation"
        time_costs = []
        # warmup
        pipe.img2img(
            prompt,
            image=init_image,
            num_inference_steps=20,
            height=height,
            width=width,
            ip_adapter_image=ip_image,
            parse_prompt_type=parse_prompt_type,
        )
        print("==> Test img2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            images = pipe.img2img(
                prompt,
                image=init_image,
                num_inference_steps=args.inference_steps,
                height=height,
                width=width,
                ip_adapter_image=init_image,
                parse_prompt_type=parse_prompt_type,
            ).images
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        images[0].save(f"{folder}/img2img.png")

    if args.task_name in ["inpaint", "inpaint_legacy", "all"]:
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
        ip_image = load_image(img_url)
        init_image = load_image(img_url)
        mask_image = load_image(mask_url)
        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        time_costs = []
        # warmup
        if args.task_name in ["inpaint_legacy", "all"]:
            call_fn = pipe.inpaint_legacy
            task_name = "inpaint_legacy"
        else:
            call_fn = pipe.inpaint
            task_name = "inpaint"
        call_fn(
            prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=20,
            height=height,
            width=width,
            ip_adapter_image=ip_image,
            parse_prompt_type=parse_prompt_type,
        )
        print(f"==> Test {task_name} performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            images = call_fn(
                prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=args.inference_steps,
                height=height,
                width=width,
                ip_adapter_image=ip_image,
                parse_prompt_type=parse_prompt_type,
            ).images
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )

        images[0].save(f"{folder}/{task_name}.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
