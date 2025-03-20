# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import paddle

import ppdiffusers

sys.path.append("fkd_diffusers")
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
from fkd_diffusers.fkd_pipeline_sd_3 import FKDStableDiffusion3Pipeline
from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_diffusers.rewards import do_eval


def main(args):
    paddle.seed(seed=args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps
    if args.use_smc:
        assert args.resample_frequency > 0
        assert args.num_particles > 1

    if "xl" in args.model_name:
        print("Using SDXL")
        pipe = FKDStableDiffusionXL.from_pretrained(args.model_name, paddle_dtype=paddle.float16)
        pipe.scheduler = ppdiffusers.DDIMScheduler.from_config(pipe.scheduler.config)
    elif "stable-diffusion-3" in args.model_name:
        print("Using SD3")
        pipe = FKDStableDiffusion3Pipeline.from_pretrained(args.model_name, paddle_dtype=paddle.float16)
    else:
        print("Using SD")
        pipe = FKDStableDiffusion.from_pretrained(args.model_name, paddle_dtype=paddle.float16)

        pipe.scheduler = ppdiffusers.DDIMScheduler.from_config(pipe.scheduler.config)

    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, cur_time)
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        import time

        print("Sleeping for a random time")
        time.sleep(np.random.randint(1, 10))
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, cur_time)
        os.makedirs(output_dir, exist_ok=False)
    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    metrics_to_compute = args.metrics_to_compute.split("#")

    metrics_arr = {metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute}

    average_time = 0
    prompt = [args.prompt] * args.num_particles

    start_time = datetime.now()

    fkd_args = dict(
        lmbda=args.lmbda,
        num_particles=args.num_particles,
        use_smc=args.use_smc,
        adaptive_resampling=args.adaptive_resampling,
        resample_frequency=args.resample_frequency,
        time_steps=args.num_inference_steps,
        resampling_t_start=args.resample_t_start,
        resampling_t_end=args.resample_t_end,
        guidance_reward_fn=args.guidance_reward_fn,
        potential_type=args.potential_type,
    )
    images = pipe(
        prompt,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        fkd_args=fkd_args,
        generator=paddle.Generator().manual_seed(args.seed),
    )
    images = images[0]

    if args.use_smc:
        end_time = datetime.now()
    results = do_eval(prompt=prompt, images=images, metrics_to_compute=metrics_to_compute)
    if not args.use_smc:
        end_time = datetime.now()
    time_taken = end_time - start_time
    results["time_taken"] = time_taken.total_seconds()
    results["prompt"] = prompt

    average_time += time_taken.total_seconds()
    print(f"Time taken: {average_time}")
    guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
    sorted_idx = np.argsort(guidance_reward)[::-1]
    images = [images[i] for i in sorted_idx]
    for metric in metrics_to_compute:
        results[metric]["result"] = [results[metric]["result"][i] for i in sorted_idx]
    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] += results[metric]["mean"]
        metrics_arr[metric]["max"] += results[metric]["max"]
        metrics_arr[metric]["min"] += results[metric]["min"]
        metrics_arr[metric]["std"] += results[metric]["std"]
    for metric in metrics_to_compute:
        print(
            metric,
            metrics_arr[metric]["mean"],
            metrics_arr[metric]["max"],
        )
    if args.save_individual_images:
        sample_path = os.path.join(output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)
        for image_idx, image in enumerate(images):
            image.save(os.path.join(sample_path, f"{image_idx:05}.png"))
        best_of_n_sample_path = os.path.join(output_dir, "best_of_n_samples")
        os.makedirs(best_of_n_sample_path, exist_ok=True)
        for image_idx, image in enumerate(images[:1]):
            image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    if args.num_particles > 1:
        _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles * 5, 5))
        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].axis("off")
        plt.suptitle(prompt[0])
        image_fpath = os.path.join(output_dir, "grid.png")
        plt.savefig(image_fpath)
        plt.close()

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_arr, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="geneval_outputs")
    parser.add_argument("--save_individual_images", type=bool, default=True)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_smc", action="store_true")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt", type=str, default=" ")

    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=5)
    parser.add_argument("--resample_t_start", type=int, default=5)
    parser.add_argument("--resample_t_end", type=int, default=30)
    parser.add_argument("--potential_type", type=str, default="diff")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    main(args)
