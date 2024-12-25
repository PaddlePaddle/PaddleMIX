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
import gc
import os
import re

import numpy as np
import paddle
from decord import VideoReader
from moviepy.editor import ImageSequenceClip
from PIL import Image

from ppdiffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXTransformer3DVCtrlModel,
    CogVideoXVCtrlPipeline,
    VCtrlModel,
)


def write_mp4(video_path, samples, fps=8):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac")


def save_vid_side_by_side(batch_output, validation_control_images, output_folder, fps):
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    ori_video_path = output_folder + "/origin_predict.mp4"
    video_path = output_folder + "/test_1.mp4"
    ori_final_images = []
    final_images = []
    outputs = []

    def get_concat_h(im1, im2):
        dst = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        ori_final_images.append(np.array(image_list[1]))
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))
    write_mp4(ori_video_path, ori_final_images, fps=fps)
    write_mp4(video_path, final_images, fps=fps)
    output_path = output_folder + "/output.mp4"
    write_mp4(output_path, outputs, fps=fps)


def load_images_from_folder_to_pil(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def frame_number(filename):
        new_pattern_match = re.search("frame_(\\d+)_7fps", filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))
        matches = re.findall("\\d+", filename)
        if matches:
            if matches[-1] == "0000" and len(matches) > 1:
                return int(matches[-2])
            return int(matches[-1])
        return float("inf")

    sorted_files = sorted(os.listdir(folder), key=frame_number)
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)
    return images


def load_images_from_video_to_pil(video_path):
    images = []
    vr = VideoReader(video_path)
    length = len(vr)
    for idx in range(length):
        frame = vr[idx].asnumpy()
        images.append(Image.fromarray(frame))
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX with VCTRL")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--control_images_folder", type=str, default=None, required=False)
    parser.add_argument("--control_video_path", type=str, default=None, required=False)
    parser.add_argument(
        "--control_mask_images_folder",
        type=str,
        default=None,
        help=("the validation mask images"),
    )
    parser.add_argument(
        "--control_mask_video_path",
        type=str,
        default=None,
        help=("the validation mask images from video"),
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--height", type=int, default=720, required=False)
    parser.add_argument("--width", type=int, default=480, required=False)
    parser.add_argument("--max_frame", type=int, default=9, required=False)
    parser.add_argument("--guidance_scale", type=float, default=3.5, required=False)
    parser.add_argument("--num_inference_steps", type=int, default=25, required=False)
    parser.add_argument("--fps", type=int, default=8, required=False)
    parser.add_argument("--vctrl_path", type=str, default=None, required=False)
    parser.add_argument("--transformer_path", type=str, default=None, required=False)
    parser.add_argument("--ref_image_path", type=str, default=None, required=False)
    parser.add_argument("--task", type=str, default=None, required=True)
    parser.add_argument("--prompt", type=str, default=None, required=True)
    parser.add_argument("--vctrl_config", type=str, default=None, help="the config file for vctrl")
    parser.add_argument(
        "--random_initialization",
        action="store_true",
        help="use random initialization instead of loading pretrained weights",
    )
    parser.add_argument("--conditioning_scale", type=float, default=1.0, required=False)
    parser.add_argument(
        "--vctrl_layout_type",
        type=str,
        default="even",
        choices=["even", "spacing", "end"],
        help="The layout type for vctrl. Choices: 'even', 'spacing', 'end'.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert (args.control_images_folder is None) ^ (
        args.control_video_path is None
    ), "must and only one of [control_images_folder, control_video_path] should be given"
    if args.control_images_folder is not None:
        validation_control_images = load_images_from_folder_to_pil(args.control_images_folder)
    else:
        validation_control_images = load_images_from_video_to_pil(args.control_video_path)

    if args.control_mask_images_folder is not None:
        validation_mask_images = load_images_from_folder_to_pil(args.control_mask_images_folder)
    elif args.control_mask_video_path is not None:
        validation_mask_images = load_images_from_video_to_pil(args.control_mask_video_path)

    if args.vctrl_path.endswith(".pdparams"):
        vctrl = VCtrlModel.from_config(args.vctrl_config)
        vctrl.set_state_dict(state_dict=paddle.load(args.vctrl_path))

    elif args.random_initialization:
        vctrl = VCtrlModel.from_config(args.vctrl_config)
    else:
        vctrl = VCtrlModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vctrl", low_cpu_mem_usage=True
        )

    if args.transformer_path:
        transformer = CogVideoXTransformer3DVCtrlModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", low_cpu_mem_usage=True
        )

        pipeline = CogVideoXVCtrlPipeline.from_pretrained(
            args.pretrained_model_name_or_path, transformer=transformer, vctrl=vctrl
        )
    else:
        pipeline = CogVideoXVCtrlPipeline.from_pretrained(
            args.pretrained_model_name_or_path, vctrl=vctrl, paddle_dtype=paddle.float16
        )

    pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    gc.collect()
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    paddle.device.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    final_result = []
    if args.task == "character_pose":
        ref_image = Image.open(args.ref_image_path).convert("RGB")
        validation_control_images = [ref_image] + validation_control_images

    num_frames = len(validation_control_images)
    num_frames = min(num_frames, args.max_frame)

    video = pipeline(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        num_frames=num_frames,
        guidance_scale=args.guidance_scale,
        generator=paddle.Generator().manual_seed(42),
        conditioning_frames=validation_control_images[:num_frames],
        conditioning_frame_indices=list(range(num_frames)),
        conditioning_scale=args.conditioning_scale,
        width=args.width,
        height=args.height,
        task=args.task,
        conditioning_masks=validation_mask_images[:num_frames] if args.task == "mask" else None,
        vctrl_layout_type=args.vctrl_layout_type,
    ).frames[0]

    final_result.append(video)
    save_vid_side_by_side(final_result, validation_control_images[:num_frames], args.output_dir, fps=args.fps)
