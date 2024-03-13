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

import paddle

# isort: split
import numpy as np
from paddlenlp.trainer.argparser import strtobool
from tqdm.auto import trange

from ppdiffusers import DiffusionPipeline, StableVideoDiffusionPipeline  # noqa
from ppdiffusers.utils import load_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="stabilityai/stable-video-diffusion-img2vid-xt@paddleinfer",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=25,
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
        default="paddle_tensorrt",
        choices=["paddle", "paddle_tensorrt"],
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
            "img2video",
            "all",
        ],
        help="The task can be one of [text2img, all]. ",
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
    parser.add_argument("--height", type=int, default=576, help="Height of input image")
    parser.add_argument("--width", type=int, default=1024, help="Width of input image")
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

    pipe = StableVideoDiffusionPipeline.from_pretrained(args.model_dir, paddle_dtype=paddle.float16)
    pipe.set_progress_bar_config(disable=False)
    width = args.width
    height = args.height

    folder = f"results-{args.backend}"
    os.makedirs(folder, exist_ok=True)

    if args.task_name in ["img2video", "all"]:
        # img2video
        img_url = "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/rocket.png"
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
            paddle.seed(seed)
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
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        frames[0][0].save(f"{folder}/test_svd.gif", save_all=True, append_images=frames[0][1:], loop=0)


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
