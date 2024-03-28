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
import paddle
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger
from tqdm.auto import trange

from ppdiffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from ppdiffusers.utils import load_image

logger.set_level("WARNING")


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="The model directory of diffusion_model.",
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
        default="sdxl_models",
        help="Path to the `ppdiffusers`'s ip-adapter checkpoint to convert (either a local directory or on the bos).Example: models",
    )
    parser.add_argument(
        "--ipadapter_weight_name",
        type=str,
        default="ip-adapter_sdxl.safetensors",
        help="Name of the weight to convert.Example: ip-adapter_sd15.safetensors",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text2img",
        choices=["text2img", "text2img_with_refiner", "img2img", "inpainting"],
        help="task.",
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=1, help="The number of performance benchmark steps.")
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument(
        "--attention_type", type=str, default="raw", choices=["raw", "cutlass", "flash", "all"], help="attention_type."
    )
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument("--height", type=int, default=512, help="Height of input image")
    parser.add_argument("--width", type=int, default=512, help="Width of input image")
    return parser.parse_args()


def text2img(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    seed = 1024
    paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-base-1.0",
        paddle_dtype=paddle_dtype,
    )
    pipe.load_ip_adapter(
        args.ipadapter_pretrained_model_name_or_path,
        subfolder=args.ipadapter_model_subfolder,
        weight_name=args.ipadapter_weight_name,
    )
    pipe.set_progress_bar_config(disable=False)
    if args.attention_type == "all":
        args.attention_type = ["raw", "cutlass", "flash"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()
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

        # width = args.width
        # height = args.height
        folder = f"attn_{attention_type}_fp16" if args.use_fp16 else f"attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # text2img
        time_costs = []
        # warmup
        img_url = "https://paddlenlp.bj.bcebos.com/models/community/examples/images/load_neg_embed.png"
        ip_image = load_image(img_url)
        prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
        negative_prompt = "text, watermark"
        image = pipe(
            prompt, negative_prompt=negative_prompt, ip_adapter_image=ip_image, num_inference_steps=50
        ).images[0]

        print("==> Test text2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            image = pipe(
                prompt, negative_prompt=negative_prompt, ip_adapter_image=ip_image, num_inference_steps=50
            ).images[0]
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/text2img.png")


def text2img_with_refiner(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    seed = 1024
    paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    base = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-base-1.0",
        paddle_dtype=paddle_dtype,
    )
    base.load_ip_adapter(
        args.ipadapter_pretrained_model_name_or_path,
        subfolder=args.ipadapter_model_subfolder,
        weight_name=args.ipadapter_weight_name,
    )
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        paddle_dtype=paddle_dtype,
    )
    base.set_progress_bar_config(disable=False)
    refiner.set_progress_bar_config(disable=False)
    if args.attention_type == "all":
        args.attention_type = ["raw", "cutlass", "flash"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        if attention_type == "raw":
            base.unet.set_default_attn_processor()
            base.vae.set_default_attn_processor()
            refiner.unet.set_default_attn_processor()
            refiner.vae.set_default_attn_processor()
        else:
            try:
                base.enable_xformers_memory_efficient_attention(attention_type)
                refiner.enable_xformers_memory_efficient_attention(attention_type)
            except Exception as e:
                if attention_type == "flash":
                    warnings.warn(
                        "Attention type flash is not supported on your GPU! We need to use 3060、3070、3080、3090、4060、4070、4080、4090、A30、A100 etc."
                    )
                    continue
                else:
                    raise ValueError(e)

        # width = args.width
        # height = args.height
        folder = f"attn_{attention_type}_fp16" if args.use_fp16 else f"attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # text2img_with_refiner
        time_costs = []
        # warmup
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        # n_steps = 40
        # high_noise_frac = 0.8
        prompt = "A majestic lion jumping from a big stone at night"
        prompt = "a photo of an astronaut riding a horse on mars"
        img_url = "https://paddlenlp.bj.bcebos.com/models/community/examples/images/load_neg_embed.png"
        ip_image = load_image(img_url)
        # run both experts
        image = base(
            prompt=prompt,
            ip_adapter_image=ip_image,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            image=image,
        ).images[0]

        print("==> Test text2img_with_refiner performance.")
        ip_image = load_image(image)
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            image = base(
                prompt=prompt,
                ip_adapter_image=ip_image,
                output_type="latent",
            ).images
            image = refiner(
                prompt=prompt,
                image=image,
            ).images[0]
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/text2img_with_refiner.png")


def img2img(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    seed = 1024
    paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-refiner-1.0",
        paddle_dtype=paddle_dtype,
    )
    pipe.load_ip_adapter(
        args.ipadapter_pretrained_model_name_or_path,
        subfolder=args.ipadapter_model_subfolder,
        weight_name=args.ipadapter_weight_name,
    )
    pipe.set_progress_bar_config(disable=False)
    if args.attention_type == "all":
        args.attention_type = ["raw", "cutlass", "flash"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()
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

        # width = args.width
        # height = args.height
        folder = f"attn_{attention_type}_fp16" if args.use_fp16 else f"attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # img2img
        time_costs = []
        # warmup
        url = "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-0-19-3/000000009.png"
        init_image = load_image(url).convert("RGB")
        ip_image = load_image(url)
        prompt = "a photo of an astronaut riding a horse on mars"
        image = pipe(prompt, ip_adapter_image=ip_image, image=init_image).images[0]

        print("==> Test img2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            image = pipe(prompt, ip_adapter_image=ip_image, image=init_image).images[0]
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/img2img.png")


def inpainting(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    seed = 1024
    paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-base-1.0",
        paddle_dtype=paddle_dtype,
        variant="fp16",
    )
    pipe.load_ip_adapter(
        args.ipadapter_pretrained_model_name_or_path,
        subfolder=args.ipadapter_model_subfolder,
        weight_name=args.ipadapter_weight_name,
    )
    pipe.set_progress_bar_config(disable=False)
    if args.attention_type == "all":
        args.attention_type = ["raw", "cutlass", "flash"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()
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

        # width = args.width
        # height = args.height
        folder = f"attn_{attention_type}_fp16" if args.use_fp16 else f"attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # inpainting
        time_costs = []
        # warmup
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        init_image = load_image(img_url).convert("RGB")
        mask_image = load_image(mask_url).convert("RGB")
        ip_image = load_image(img_url)
        prompt = "A majestic tiger sitting on a bench"
        image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            ip_adapter_image=ip_image,
            num_inference_steps=50,
            strength=0.80,
        ).images[0]

        print("==> Test inpainting performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            image = pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                ip_adapter_image=ip_image,
                num_inference_steps=50,
                strength=0.80,
            ).images[0]
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/inpainting.png")


if __name__ == "__main__":
    args = parse_arguments()
    if args.task == "text2img":
        text2img(args)
    elif args.task == "text2img_with_refiner":
        text2img_with_refiner(args)
    elif args.task == "img2img":
        img2img(args)
    elif args.task == "inpainting":
        inpainting(args)
