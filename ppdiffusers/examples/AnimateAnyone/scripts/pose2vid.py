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
from datetime import datetime
from pathlib import Path

import paddle
from einops import repeat
from omegaconf import OmegaConf
from paddle.vision import transforms
from paddlenlp.transformers import CLIPVisionModelWithProjection
from PIL import Image
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_video_as_mp4

from ppdiffusers import AutoencoderKL, DDIMScheduler
from ppdiffusers.models.AnimateAnyone.pose_guider import PoseGuider
from ppdiffusers.models.AnimateAnyone.unet_2d_condition import UNet2DConditionModel
from ppdiffusers.models.AnimateAnyone.unet_3d import UNet3DConditionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=784)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = paddle.float16
    else:
        weight_dtype = paddle.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
        paddle_dtype=weight_dtype,
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        paddle_dtype=weight_dtype,
        subfolder="unet",
    )

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        denoising_unet_config_path=config.denoising_unet_config_path,
        base_model_path=config.denoising_unet_path,
        motion_module_path=config.motion_module_path,
        weight_dtype=weight_dtype,
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256), weight_dtype=weight_dtype)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path,
        dtype=weight_dtype,
    )

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = paddle.Generator().manual_seed(args.seed)

    width, height = args.W, args.H

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe.load_pretrained(config)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    for ref_image_path in config["test_cases"].keys():
        # Each ref_image may correspond to multiple actions
        for pose_video_path in config["test_cases"][ref_image_path]:
            ref_name = Path(ref_image_path).stem
            pose_name = Path(pose_video_path).stem.replace("_kps", "")

            ref_image_pil = Image.open(ref_image_path).convert("RGB")

            pose_list = []
            pose_tensor_list = []
            pose_images = read_frames(pose_video_path)
            src_fps = get_fps(pose_video_path)
            print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
            pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
            for pose_image_pil in pose_images[: args.L]:
                pose_tensor_list.append(pose_transform(pose_image_pil))
                pose_list.append(pose_image_pil)

            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
            ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=args.L)

            pose_tensor = paddle.stack(x=pose_tensor_list, axis=0)

            pose_tensor = pose_tensor.transpose([1, 0, 2, 3])
            pose_tensor = pose_tensor.unsqueeze(axis=0)

            video = pipe(
                ref_image_pil,
                pose_list,
                width,
                height,
                args.L,
                args.steps,
                args.cfg,
                generator=generator,
            ).videos

            save_video_as_mp4(
                video,
                f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
                fps=src_fps if args.fps is None else args.fps,
            )


if __name__ == "__main__":
    main()
