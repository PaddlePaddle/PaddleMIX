# !pip install opencv-python transformers accelerate
import os
import argparse
# import torch
import paddle
import math
import numpy as np
from PIL import Image
from ppdiffusers import StableDiffusion3ControlNetPipeline
from ppdiffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from ppdiffusers.models import SD3ControlNetModel

def controlnet_process(image_path, prompt, task, mask=None, reverse_mask=False, controlnet_seed=0, controlnet_num_inference_steps=28, controlnet_guidance_scale=7.0, controlnet_conditioning_scale=1.0):
    output_image_path = image_path.replace('.jpg', '_controlnet.jpg').replace('.png', '_controlnet.png')
    image = Image.open(image_path)
    if mask is not None:
        mask = Image.open(mask)
        if reverse_mask:
            print("note that reverse the mask")
            mask = np.array(mask)
            zero_mask = (mask == 0)
            mask[mask == 255] = 0
            mask[zero_mask] = 255
            mask = Image.fromarray(mask)
    width, height = image.size
    output_width, output_height = math.ceil(width/64)*64, math.ceil(height/64)*64

    if task == "canny":
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", paddle_dtype=paddle.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16
        )
    elif task == "pose":
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose", paddle_dtype=paddle.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16
        )
    elif task == "mask":
        controlnet = SD3ControlNetModel.from_pretrained("alimama-creative/SD3-Controlnet-Inpainting", paddle_dtype=paddle.float16)
        pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16)

    paddle.seed(seed=controlnet_seed)
    pipe.set_progress_bar_config(disable=None)

    if task == "mask":
        image = pipe(
            prompt, control_image=image, control_mask=mask,
            width=output_width, height=output_height,
            num_inference_steps=controlnet_num_inference_steps,
            guidance_scale=controlnet_guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images[0]
    else:
        image = pipe(
            prompt, control_image=image,
            width=output_width, height=output_height,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=controlnet_guidance_scale,
            num_inference_steps=controlnet_num_inference_steps,
        ).images[0]
    image.save(output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("controlnet-SD3", add_help=True)
    parser.add_argument("--image_path", type=str,default=None)
    parser.add_argument("--prompt", type=str,default=None)
    parser.add_argument("--task", type=str,default=None)
    parser.add_argument("--mask_path", type=str,default=None)
    parser.add_argument("--reverse_mask", type=bool,default=False)
    parser.add_argument("--controlnet_seed", type=int,default=0)
    parser.add_argument("--controlnet_num_inference_steps", type=int,default=28)
    parser.add_argument("--controlnet_guidance_scale", type=float,default=7.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float,default=1.0)
    args = parser.parse_args()
    controlnet_process(
        args.image_path,
        args.prompt,
        args.task,
        args.mask_path,
        args.reverse_mask,
        args.controlnet_seed,
        args.controlnet_num_inference_steps,
        args.controlnet_guidance_scale,
        args.controlnet_conditioning_scale)