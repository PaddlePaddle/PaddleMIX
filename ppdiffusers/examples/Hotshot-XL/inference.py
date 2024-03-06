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

import argparse
import os
import sys
from contextlib import contextmanager

import paddle
from einops import rearrange

sys.path.append(".")
from hotshot_xl import vision_utils
from hotshot_xl.utils import (
    extract_gif_frames_from_midpoint,
    save_as_gif,
    save_as_mp4,
    scale_aspect_fill,
)

import ppdiffusers
from ppdiffusers.models.hotshot_xl.unet import UNet3DConditionModel
from ppdiffusers.pipelines.hotshot_xl.hotshot_xl_controlnet_pipeline import (
    HotshotXLControlNetPipeline,
)
from ppdiffusers.pipelines.hotshot_xl.hotshot_xl_pipeline import HotshotXLPipeline

SCHEDULERS = {
    "EulerAncestralDiscreteScheduler": ppdiffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
    "EulerDiscreteScheduler": ppdiffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
    "default": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Hotshot-XL inference")
    parser.add_argument("--pretrained_path", type=str, default="co63oc/hotshotxl/hotshot_output")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--spatial_unet_base", type=str)
    parser.add_argument("--lora", type=str)
    parser.add_argument("--output", type=str, default="output.gif")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--prompt", type=str, default="a bulldog in the captains chair of a spaceship, hd, high quality"
    )
    parser.add_argument("--negative_prompt", type=str, default="blurry")
    parser.add_argument("--seed", type=int, default=455)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--target_width", type=int, default=512)
    parser.add_argument("--target_height", type=int, default=512)
    parser.add_argument("--og_width", type=int, default=1920)
    parser.add_argument("--og_height", type=int, default=1080)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--video_duration", type=int, default=1000)
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument(
        "--scheduler", type=str, default="EulerAncestralDiscreteScheduler", help="Name of the scheduler to use"
    )
    parser.add_argument("--control_type", type=str, default=None, choices=["depth", "canny"])
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.7)
    parser.add_argument("--control_guidance_start", type=float, default=0.0)
    parser.add_argument("--control_guidance_end", type=float, default=1.0)
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--precision", type=str, default="f32", choices=["f16", "f32", "bf16"])
    parser.add_argument("--autocast", type=str, default=None, choices=["f16", "bf16"])
    return parser.parse_args()


to_pil = vision_utils.ToPILImage()


def to_pil_images(video_frames: paddle.Tensor, output_type="pil"):
    video_frames = rearrange(video_frames, "b c f w h -> b f c w h")
    bsz = tuple(video_frames.shape)[0]
    images = []
    for i in range(bsz):
        video = video_frames[i]
        for j in range(tuple(video.shape)[0]):
            if output_type == "pil":
                images.append(to_pil(video[j]))
            else:
                images.append(video[j])
    return images


@contextmanager
def maybe_auto_cast(data_type):
    if data_type:
        with paddle.amp.auto_cast(dtype=data_type):
            yield
    else:
        yield


def main():
    args = parse_args()
    if args.control_type and not args.gif:
        raise ValueError("Controlnet specified but you didn't specify a gif!")
    if args.gif and not args.control_type:
        print("warning: gif was specified but no control type was specified. gif will be ignored.")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    device = paddle.device.get_device()
    control_net_model_pretrained_path = None
    if args.control_type:
        control_type_to_model_map = {
            # "canny": "diffusers/controlnet-canny-sdxl-1.0",
            # "depth": "diffusers/controlnet-depth-sdxl-1.0",
            "canny": "co63oc/hotshotxl/controlnet_canny",
            "depth": "co63oc/hotshotxl/controlnet_depth",
        }
        control_net_model_pretrained_path = control_type_to_model_map[args.control_type]
    data_type = "float32"
    if args.precision == "f16":
        data_type = "float16"
    elif args.precision == "f32":
        data_type = "float32"
    elif args.precision == "bf16":
        data_type = "bfloat16"
    pipe_line_args = {"paddle_dtype": data_type, "use_safetensors": True}
    PipelineClass = HotshotXLPipeline
    if control_net_model_pretrained_path:
        PipelineClass = HotshotXLControlNetPipeline
        pipe_line_args["controlnet"] = ppdiffusers.ControlNetModel.from_pretrained(
            control_net_model_pretrained_path, paddle_dtype=data_type
        )
    if args.spatial_unet_base:
        unet_3d = UNet3DConditionModel.from_pretrained(
            args.pretrained_path, subfolder="unet", paddle_dtype=data_type
        ).to(device)
        unet = UNet3DConditionModel.from_pretrained_spatial(args.spatial_unet_base).to(device, dtype=data_type)
        temporal_layers = {}
        unet_3d_sd = unet_3d.state_dict()
        for k, v in unet_3d_sd.items():
            if "temporal" in k:
                temporal_layers[k] = v
        unet.set_state_dict(state_dict=temporal_layers, use_structured_name=False)
        pipe_line_args["unet"] = unet
        del unet_3d_sd
        del unet_3d
        del temporal_layers
    pipe = PipelineClass.from_pretrained(args.pretrained_path, **pipe_line_args).to(device)
    if args.lora:
        pipe.load_lora_weights(args.lora)
    SchedulerClass = SCHEDULERS[args.scheduler]
    if SchedulerClass is not None:
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
    if args.xformers:
        pipe.enable_xformers_memory_efficient_attention()
    generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
    autocast_type = None
    if args.autocast == "f16":
        autocast_type = "float16"
    elif args.autocast == "bf16":
        autocast_type = "bfloat16"
    if type(pipe) is HotshotXLControlNetPipeline:
        kwargs = {}
    else:
        kwargs = {"low_vram_mode": args.low_vram_mode}
    if args.gif and type(pipe) is HotshotXLControlNetPipeline:
        kwargs["control_images"] = [
            scale_aspect_fill(img, args.width, args.height).convert("RGB")
            for img in extract_gif_frames_from_midpoint(
                args.gif, fps=args.video_length, target_duration=args.video_duration
            )
        ]
        kwargs["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale
        kwargs["control_guidance_start"] = args.control_guidance_start
        kwargs["control_guidance_end"] = args.control_guidance_end
    with maybe_auto_cast(autocast_type):
        images = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            original_size=(args.og_width, args.og_height),
            target_size=(args.target_width, args.target_height),
            num_inference_steps=args.steps,
            video_length=args.video_length,
            generator=generator,
            output_type="tensor",
            **kwargs,
        ).videos
    images = to_pil_images(images, output_type="pil")
    if args.video_length > 1:
        if args.output.split(".")[-1] == "gif":
            save_as_gif(images, args.output, duration=args.video_duration // args.video_length)
        else:
            save_as_mp4(images, args.output, duration=args.video_duration // args.video_length)
    else:
        images[0].save(args.output, format="JPEG", quality=95)


if __name__ == "__main__":
    main()
