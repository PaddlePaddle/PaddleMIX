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

# torch.nn.functional.scaled_dot_product_attention_ = torch.nn.functional.scaled_dot_product_attention
# delattr(torch.nn.functional, "scaled_dot_product_attention")
import numpy as np
import PIL
import requests
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLInstructPix2PixPipeline,
    UniPCMultistepScheduler,
)

# from diffusers.models.attention_processor import AttnProcessor,AttnProcessor2_0
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
        "--task",
        type=str,
        default="text2img",
        choices=["text2img", "text2img_with_refiner", "img2img", "inpainting", "instruct_pix2pix"],
        help="task.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="The model directory of diffusion_model.",
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=10, help="The number of performance benchmark steps.")
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
    parser.add_argument("--channels_last", type=strtobool, default=False, help="Whether to use channels_last")
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Whether to use FP16 mode")
    parser.add_argument("--tf32", type=strtobool, default=True, help="tf32")
    parser.add_argument("--compile", type=strtobool, default=False, help="compile")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="sdp",
        choices=[
            "raw",
            "sdp",
        ],
        help="attention_type.",
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
    return parser.parse_args()


def attn_processors(self):
    processors = {}

    def fn_recursive_add_processors(name: str, module, processors):
        if hasattr(module, "set_processor"):
            processors[f"{name}.processor"] = module.processor

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in self.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def set_attn_processor(self, processor):
    count = len(attn_processors(self).keys())

    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
            f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        )

    def fn_recursive_attn_processor(name: str, module, processor):
        if hasattr(module, "set_processor"):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in self.named_children():
        fn_recursive_attn_processor(name, module, processor)


def upcast_vae(self, target_type=torch.float16):
    self.vae.to(dtype=target_type)


def text2img(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
    )
    scheduler = change_scheduler(pipe, args.scheduler)
    pipe.scheduler = scheduler
    if args.device_id >= 0:
        pipe.to(f"cuda:{args.device_id}")

    if args.attention_type == "all":
        args.attention_type = ["raw", "sdp"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        # attn_prrocessor_cls = AttnProcessor if attention_type == "raw" else AttnProcessor2_0
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()
        # set_attn_processor(pipe.unet, attn_prrocessor_cls())
        # set_attn_processor(pipe.vae, attn_prrocessor_cls())

        if args.channels_last:
            pipe.unet.to(memory_format=torch.channels_last)

        if args.compile:
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        # width = args.width
        # height = args.height
        pipe.set_progress_bar_config(disable=True)

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # text2img
        time_costs = []
        # warmup
        prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
        negative_prompt = "text, watermark"
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
        upcast_vae(pipe, target_type=torch_dtype)

        print("==> Test text2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            torch.cuda.manual_seed(seed)
            image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
            upcast_vae(pipe, target_type=torch_dtype)
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/text2img_torch.png")


def text2img_with_refiner(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    base = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
    )
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch_dtype,
    )
    scheduler = change_scheduler(base, args.scheduler)
    base.scheduler = scheduler
    refiner.scheduler = scheduler
    if args.device_id >= 0:
        base.to(f"cuda:{args.device_id}")
        refiner.to(f"cuda:{args.device_id}")

    if args.attention_type == "all":
        args.attention_type = ["raw", "sdp"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        # attn_prrocessor_cls = AttnProcessor if attention_type == "raw" else AttnProcessor2_0
        if attention_type == "raw":
            base.unet.set_default_attn_processor()
            base.vae.set_default_attn_processor()
            refiner.unet.set_default_attn_processor()
            refiner.vae.set_default_attn_processor()

        if args.channels_last:
            base.unet.to(memory_format=torch.channels_last)
            refiner.unet.to(memory_format=torch.channels_last)

        if args.compile:
            print("Run torch compile")
            base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

        # width = args.width
        # height = args.height
        base.set_progress_bar_config(disable=True)
        refiner.set_progress_bar_config(disable=True)

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # text2img_with_refiner
        time_costs = []
        # warmup
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        # n_steps = 40
        # high_noise_frac = 0.8
        prompt = "A majestic lion jumping from a big stone at night"
        prompt = "a photo of an astronaut riding a horse on mars"
        # run both experts
        image = base(
            prompt=prompt,
            output_type="latent",
        ).images
        upcast_vae(base, target_type=torch_dtype)
        image = refiner(
            prompt=prompt,
            image=image,
        ).images[0]
        upcast_vae(base, target_type=torch_dtype)
        upcast_vae(refiner, target_type=torch_dtype)

        print("==> Test text2img_with_refiner performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            torch.cuda.manual_seed(seed)
            image = base(
                prompt=prompt,
                output_type="latent",
            ).images
            upcast_vae(base, target_type=torch_dtype)
            image = refiner(
                prompt=prompt,
                image=image,
            ).images[0]
            upcast_vae(base, target_type=torch_dtype)
            upcast_vae(refiner, target_type=torch_dtype)
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/text2img_with_refiner_torch.png")


def img2img(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch_dtype,
    )
    scheduler = change_scheduler(pipe, args.scheduler)
    pipe.scheduler = scheduler
    if args.device_id >= 0:
        pipe.to(f"cuda:{args.device_id}")

    if args.attention_type == "all":
        args.attention_type = ["raw", "sdp"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        # attn_prrocessor_cls = AttnProcessor if attention_type == "raw" else AttnProcessor2_0
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()

        if args.channels_last:
            pipe.unet.to(memory_format=torch.channels_last)

        if args.compile:
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        # width = args.width
        # height = args.height
        pipe.set_progress_bar_config(disable=True)

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # img2img
        time_costs = []
        # warmup
        url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
        init_image = load_image(url).convert("RGB")
        prompt = "a photo of an astronaut riding a horse on mars"
        image = pipe(prompt, image=init_image).images[0]
        upcast_vae(pipe, target_type=torch_dtype)

        print("==> Test img2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            torch.cuda.manual_seed(seed)
            image = pipe(prompt, image=init_image).images[0]
            upcast_vae(pipe, target_type=torch_dtype)
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/img2img_torch.png")


def inpainting(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
    )
    scheduler = change_scheduler(pipe, args.scheduler)
    pipe.scheduler = scheduler
    if args.device_id >= 0:
        pipe.to(f"cuda:{args.device_id}")

    if args.attention_type == "all":
        args.attention_type = ["raw", "sdp"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        # attn_prrocessor_cls = AttnProcessor if attention_type == "raw" else AttnProcessor2_0
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()

        if args.channels_last:
            pipe.unet.to(memory_format=torch.channels_last)

        if args.compile:
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        # width = args.width
        # height = args.height
        pipe.set_progress_bar_config(disable=True)

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # inpainting
        time_costs = []
        # warmup
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        init_image = load_image(img_url).convert("RGB")
        mask_image = load_image(mask_url).convert("RGB")
        prompt = "A majestic tiger sitting on a bench"
        image = pipe(
            prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80
        ).images[0]
        upcast_vae(pipe, target_type=torch_dtype)

        print("==> Test inpainting performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            torch.cuda.manual_seed(seed)
            image = pipe(
                prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80
            ).images[0]
            upcast_vae(pipe, target_type=torch_dtype)
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/inpainting_torch.png")


def instruct_pix2pix(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path
        else "sayakpaul/sdxl-instructpix2pix-1024-orig",
        torch_dtype=torch_dtype,
    )
    scheduler = change_scheduler(pipe, args.scheduler)
    pipe.scheduler = scheduler
    if args.device_id >= 0:
        pipe.to(f"cuda:{args.device_id}")

    if args.attention_type == "all":
        args.attention_type = ["raw", "sdp"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        # attn_prrocessor_cls = AttnProcessor if attention_type == "raw" else AttnProcessor2_0
        if attention_type == "raw":
            pipe.unet.set_default_attn_processor()
            pipe.vae.set_default_attn_processor()

        if args.channels_last:
            pipe.unet.to(memory_format=torch.channels_last)

        if args.compile:
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        # width = args.width
        # height = args.height
        pipe.set_progress_bar_config(disable=True)

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # instruct_pix2pix
        time_costs = []
        # warmup
        url = "https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg"

        def download_image(url):
            image = PIL.Image.open(requests.get(url, stream=True).raw)
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image

        image = download_image(url)
        prompt = "make it Japan"
        num_inference_steps = 20
        image_guidance_scale = 1.5
        guidance_scale = 10
        image = pipe(
            prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        ).images[0]
        upcast_vae(pipe, target_type=torch_dtype)

        print("==> Test instruct_pix2pix performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            torch.cuda.manual_seed(seed)
            image = pipe(
                prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
            ).images[0]
            upcast_vae(pipe, target_type=torch_dtype)
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        image.save(f"{folder}/instruct_pix2pix_torch.png")


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
    elif args.task == "instruct_pix2pix":
        instruct_pix2pix(args)
