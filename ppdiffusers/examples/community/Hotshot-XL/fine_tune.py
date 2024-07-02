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

import sys

sys.path.append(".")
import argparse
import gc
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import paddle
from einops import rearrange
from hotshot_xl import vision_utils
from hotshot_xl.utils import get_crop_coordinates, res_to_aspect_map, scale_aspect_fill
from PIL import Image
from tqdm.auto import tqdm  # noqa: *

import ppdiffusers
from ppdiffusers.models.hotshot_xl.unet import UNet3DConditionModel
from ppdiffusers.pipelines.hotshot_xl.hotshot_xl_pipeline import HotshotXLPipeline
from ppdiffusers.transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from ppdiffusers.utils import logging

if ppdiffusers.utils.is_wandb_available():
    import wandb  # noqa: *
logger = logging.get_logger(__file__)
from ppdiffusers import accelerate


class HotshotXLDataset(paddle.io.Dataset):
    def __init__(self, directory: str, make_sample_fn: Callable):
        """

        Training data folder needs to look like:
        + training_samples
        --- + sample_001
        ------- + frame_0.jpg
        ------- + frame_1.jpg
        ------- + ...
        ------- + frame_n.jpg
        ------- + prompt.txt
        --- + sample_002
        ------- + frame_0.jpg
        ------- + frame_1.jpg
        ------- + ...
        ------- + frame_n.jpg
        ------- + prompt.txt

        Args:
            directory: base directory of the training samples
            make_sample_fn: a delegate call to load the images and prep the sample for batching
        """
        samples_dir = [os.path.join(directory, p) for p in os.listdir(directory)]
        samples_dir = [p for p in samples_dir if os.path.isdir(p)]
        samples = []
        for d in samples_dir:
            file_paths = [os.path.join(d, p) for p in os.listdir(d)]
            image_fps = [f for f in file_paths if os.path.splitext(f)[1] in {".png", ".jpg"}]
            with open(os.path.join(d, "prompt.txt")) as f:
                prompt = f.read().strip()
            samples.append({"image_fps": image_fps, "prompt": prompt})
        self.samples = samples
        self.length = len(samples)
        self.make_sample_fn = make_sample_fn

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.make_sample_fn(self.samples[index])


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="hotshotco/Hotshot-XL",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_resume_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data to train.")
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.',
    )
    parser.add_argument("--run_validation_at_start", action="store_true")
    parser.add_argument("--max_vae_encode", type=int, default=None)
    parser.add_argument("--vae_b16", action="store_true")
    parser.add_argument("--disable_optimizer_restore", action="store_true")
    parser.add_argument(
        "--latent_nan_checking", action="store_true", help="Check if latents contain nans - important if vae is f16"
    )
    parser.add_argument("--test_prompts", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="fine-tune-hotshot-xl", help="the name of the run")
    parser.add_argument("--run_name", type=str, default="run-01", help="the name of the run")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--noise_offset", type=float, default=0.05, help="The scale of noise offset.")
    parser.add_argument("--seed", type=int, default=111, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="1.75",
        choices=list(res_to_aspect_map[512].keys()),
        help="Aspect ratio to train at",
    )
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=9999999,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-06,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--validate_every_steps", type=int, default=100, help="Run inference every")
    parser.add_argument("--save_n_steps", type=int, default=100, help="Save the model every n global_steps")
    parser.add_argument(
        "--save_starting_step",
        type=int,
        default=100,
        help="The step from which it starts saving intermediary checkpoints",
    )
    parser.add_argument("--nccl_timeout", type=int, help="nccl_timeout", default=3600)
    parser.add_argument("--snr_gamma", action="store_true")
    args = parser.parse_args()
    return args


def add_time_ids(
    unet_config,
    unet_add_embedding,
    text_encoder_2: CLIPTextModelWithProjection,
    original_size: tuple,
    crops_coords_top_left: tuple,
    target_size: tuple,
    dtype: paddle.dtype,
):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    passed_add_embed_dim = (
        unet_config.addition_time_embed_dim * len(add_time_ids) + text_encoder_2.config.projection_dim
    )
    expected_add_embed_dim = unet_add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
    add_time_ids = paddle.to_tensor(data=[add_time_ids], dtype=dtype)
    return add_time_ids


def main():
    global_step = 0
    min_steps_before_validation = 0
    args = parse_args()
    next_save_iter = args.save_starting_step
    if args.save_starting_step < 1:
        next_save_iter = None
    if args.report_to == "wandb":
        if not ppdiffusers.utils.is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[accelerate.utils.dataclasses.InitProcessGroupKwargs(timeout=timedelta(args.nccl_timeout))],
    )

    def save_model_hook(models, weights, output_dir):
        nonlocal global_step
        for model in models:
            if isinstance(model, type(unet)):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerate.utils.set_seed(args.seed)
    if accelerator.is_local_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    vae = ppdiffusers.AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    optimizer_resume_path = None
    if args.unet_resume_path:
        optimizer_fp = os.path.join(args.unet_resume_path, "optimizer.bin")
        if os.path.exists(optimizer_fp):
            optimizer_resume_path = optimizer_fp
        unet = UNet3DConditionModel.from_pretrained(
            args.unet_resume_path, subfolder="unet", low_cpu_mem_usage=False, device_map=None
        )
    else:
        unet = UNet3DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    if args.xformers:
        vae.set_use_memory_efficient_attention_xformers(True, None)
        unet.set_use_memory_efficient_attention_xformers(True, None)
    unet_config = unet.config
    unet_add_embedding = unet.add_embedding
    unet.stop_gradient = not False

    temporal_params = unet.temporal_parameters()
    for p in temporal_params:
        p.stop_gradient = not True

    vae.stop_gradient = not False
    text_encoder.stop_gradient = not False
    text_encoder_2.stop_gradient = not False

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.scale_lr:
        num_processes = 1
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
        )
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = paddle.optimizer.AdamW
    learning_rate = args.learning_rate
    params_to_optimize = [{"params": temporal_params, "learning_rate": learning_rate}]
    optimizer = optimizer_class(
        parameters=params_to_optimize,
        learning_rate=args.learning_rate,
        # betas=(args.adam_beta1, args.adam_beta2),
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
    )
    if optimizer_resume_path and not args.disable_optimizer_restore:
        logger.info("Restoring the optimizer.")
        try:
            old_optimizer_state_dict = paddle.load(path=optimizer_resume_path)
            old_state = old_optimizer_state_dict["state"]
            optimizer.set_state_dict(state_dict={"state": old_state, "param_groups": optimizer.param_groups})
            del old_optimizer_state_dict
            del old_state
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()
            gc.collect()
            logger.info("Restored the optimizer ok")
        except:
            logger.error("Failed to restore the optimizer...", exc_info=True)
            traceback.print_exc()
            raise
    noise_scheduler = ppdiffusers.DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.place)[timesteps].astype(dtype="float32")
        while len(tuple(sqrt_alphas_cumprod.shape)) < len(tuple(timesteps.shape)):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(shape=tuple(timesteps.shape))
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.place)[timesteps].astype(
            dtype="float32"
        )
        while len(tuple(sqrt_one_minus_alphas_cumprod.shape)) < len(tuple(timesteps.shape)):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(shape=tuple(timesteps.shape))
        snr = (alpha / sigma) ** 2
        return snr

    device = str("cuda").replace("cuda", "gpu")
    image_transforms = paddle.vision.transforms.Compose(
        [paddle.vision.transforms.ToTensor(), paddle.vision.transforms.Normalize([0.5], [0.5])]
    )

    def image_to_tensor(img):
        with paddle.no_grad():
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = image_transforms(img)  # .to(accelerator.place)
            if tuple(image.shape)[0] == 1:
                image = image.repeat(3, 1, 1)
            if tuple(image.shape)[0] > 3:
                image = image[:3, :, :]
        return image

    def make_sample(sample):
        nonlocal unet_config
        nonlocal unet_add_embedding
        images = [Image.open(img) for img in sample["image_fps"]]
        og_size = images[0].size
        for i, im in enumerate(images):
            if im.mode != "RGB":
                images[i] = im.convert("RGB")
        aspect_ratio_map = res_to_aspect_map[args.resolution]
        required_size = tuple(aspect_ratio_map[args.aspect_ratio])
        if required_size != og_size:

            def resize_image(x):
                img_size = x.size
                if img_size == required_size:
                    return x.resize(required_size, Image.LANCZOS)
                return scale_aspect_fill(x, required_size[0], required_size[1])

            with ThreadPoolExecutor(max_workers=len(images)) as executor:
                images = list(executor.map(resize_image, images))
        frames = paddle.stack(x=[image_to_tensor(x) for x in images])
        l, u, *_ = get_crop_coordinates(og_size, images[0].size)
        crop_coords = l, u
        additional_time_ids = add_time_ids(
            unet_config,
            unet_add_embedding,
            text_encoder_2,
            og_size,
            crop_coords,
            (required_size[0], required_size[1]),
            dtype="float32",
        ).to(device)
        input_ids_0 = tokenizer(
            sample["prompt"], padding="do_not_pad", truncation=True, max_length=tokenizer.model_max_length
        ).input_ids
        input_ids_1 = tokenizer_2(
            sample["prompt"], padding="do_not_pad", truncation=True, max_length=tokenizer.model_max_length
        ).input_ids
        return {
            "frames": frames,
            "input_ids_0": input_ids_0,
            "input_ids_1": input_ids_1,
            "additional_time_ids": additional_time_ids,
        }

    def collate_fn(examples: list) -> dict:
        input_ids_0 = [example["input_ids_0"] for example in examples]
        input_ids_0 = tokenizer.pad(
            {"input_ids": input_ids_0},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pd",
        ).input_ids
        prompt_embeds_0 = text_encoder(input_ids_0.to(device), output_hidden_states=True)
        prompt_embeds_0 = prompt_embeds_0.hidden_states[-2]
        input_ids_1 = [example["input_ids_1"] for example in examples]
        input_ids_1 = tokenizer_2.pad(
            {"input_ids": input_ids_1},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pd",
        ).input_ids
        prompt_embeds = text_encoder_2(input_ids_1.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds_1 = prompt_embeds.hidden_states[-2]
        prompt_embeds = paddle.concat(x=[prompt_embeds_0, prompt_embeds_1], axis=-1)
        *_, h, w = tuple(examples[0]["frames"].shape)
        return {
            "frames": paddle.stack(x=[x["frames"] for x in examples]).astype(dtype="float32"),
            "prompt_embeds": prompt_embeds.astype(dtype="float32"),
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "additional_time_ids": paddle.stack(x=[x["additional_time_ids"] for x in examples]),
        }

    dataset = HotshotXLDataset(args.data_dir, make_sample)
    dataloader = paddle.io.DataLoader(
        dataset=dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = ppdiffusers.optimization.get_scheduler(
        args.lr_scheduler,
        # optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(unet, optimizer, lr_scheduler, dataloader)

    def to_images(video_frames: paddle.Tensor):
        to_pil = vision_utils.ToPILImage()
        video_frames = rearrange(video_frames, "b c f w h -> b f c w h")
        bsz = tuple(video_frames.shape)[0]
        images = []
        for i in range(bsz):
            video = video_frames[i]
            for j in range(tuple(video.shape)[0]):
                image = to_pil(video[j])
                images.append(image)
        return images

    def to_video_frames(images: list) -> np.ndarray:
        x = np.stack([np.asarray(img) for img in images])
        return np.transpose(x, (0, 3, 1, 2))

    def run_validation(step=0, node_index=0):
        nonlocal global_step
        nonlocal accelerator
        if args.test_prompts:
            prompts = args.test_prompts.split("|")
        else:
            prompts = [
                "a woman is lifting weights in a gym",
                "a group of people are dancing at a party",
                "a teddy bear doing the front crawl",
            ]
        paddle.device.cuda.empty_cache()
        gc.collect()
        logger.info(f"Running inference to test model at {step} steps")
        with paddle.no_grad():
            pipe = HotshotXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                vae=vae,
            )
            videos = []
            aspect_ratio_map = res_to_aspect_map[args.resolution]
            w, h = aspect_ratio_map[args.aspect_ratio]
            for prompt in prompts:
                video = pipe(
                    prompt,
                    width=w,
                    height=h,
                    original_size=(1920, 1080),
                    target_size=(args.resolution, args.resolution),
                    num_inference_steps=30,
                    video_length=8,
                    output_type="tensor",
                    generator=paddle.framework.core.default_cpu_generator().manual_seed(111),
                ).videos
                videos.append(to_images(video))
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(
                        {"validation": [wandb.Video(to_video_frames(video), fps=8, format="mp4") for video in videos]},
                        step=global_step,
                    )
            del pipe
        return

    vae.to(dtype="bfloat16" if args.vae_b16 else "float32")
    # text_encoder.to(accelerator.place)
    # text_encoder_2.to(accelerator.place)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers(args.project_name)

    def bar(prg):
        br = "|" + "█" * prg + " " * (25 - prg) + "|"
        return br

    num_processes = 1
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    latents_scaler = vae.config.scaling_factor

    def save_checkpoint():
        save_dir = Path(args.output_dir)
        save_dir = str(save_dir)
        save_dir = save_dir.replace(" ", "_")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        accelerator.save_state(save_dir)

    def save_checkpoint_and_wait():
        if accelerator.is_main_process:
            save_checkpoint()
        accelerator.wait_for_everyone()

    def save_model_and_wait():
        if accelerator.is_main_process:
            HotshotXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                vae=vae,
            ).save_pretrained(args.output_dir, safe_serialization=True)
        accelerator.wait_for_everyone()

    def compute_loss_from_batch(batch: dict):
        frames = batch["frames"]
        bsz, number_of_frames, c, w, h = tuple(frames.shape)
        with paddle.no_grad():
            if args.max_vae_encode:
                latents = []
                x = rearrange(frames, "bs nf c h w -> (bs nf) c h w")
                for latent_index in range(0, tuple(x.shape)[0], args.max_vae_encode):
                    sample = x[latent_index : latent_index + args.max_vae_encode]
                    latent = vae.encode(sample.to(dtype=vae.dtype)).latent_dist.sample().astype(dtype="float32")
                    if len(tuple(latent.shape)) == 3:
                        latent = latent.unsqueeze(axis=0)
                    latents.append(latent)
                    paddle.device.cuda.empty_cache()
                latents = paddle.concat(x=latents, axis=0)
            else:
                x = rearrange(frames, "bs nf c h w -> (bs nf) c h w")
                del frames
                paddle.device.cuda.empty_cache()
                latents = vae.encode(x.to(dtype=vae.dtype)).latent_dist.sample().astype(dtype="float32")
            if args.latent_nan_checking and paddle.any(x=paddle.isnan(x=latents)):
                accelerator.print("NaN found in latents, replacing with zeros")
                latents = paddle.where(condition=paddle.isnan(x=latents), x=paddle.zeros_like(x=latents), y=latents)
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=bsz)
            paddle.device.cuda.empty_cache()
            noise = paddle.randn(shape=latents.shape, dtype=latents.dtype)
            if args.noise_offset:
                noise += args.noise_offset * paddle.randn(
                    shape=(tuple(latents.shape)[0], tuple(latents.shape)[1], 1, 1, 1)
                )
            timesteps = paddle.randint(low=0, high=noise_scheduler.config.num_train_timesteps, shape=(bsz,))
            timesteps = timesteps.astype(dtype="int64")
            latents = latents * latents_scaler
            prompt_embeds = batch["prompt_embeds"]
            add_text_embeds = batch["pooled_prompt_embeds"]
            additional_time_ids = batch["additional_time_ids"]
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": additional_time_ids}
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        noisy_latents.stop_gradient = not True
        model_pred = unet(
            noisy_latents,
            timesteps,
            cross_attention_kwargs=None,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        if args.snr_gamma:
            snr = compute_snr(timesteps)
            mse_loss_weights = (
                paddle.stack(x=[snr, args.snr_gamma * paddle.ones_like(x=timesteps)], axis=1).min(dim=1)[0] / snr
            )
            loss = paddle.nn.functional.mse_loss(
                input=model_pred.astype(dtype="float32"), label=target.astype(dtype="float32"), reduction="none"
            )
            loss = loss.mean(axis=list(range(1, len(tuple(loss.shape))))) * mse_loss_weights
            return loss.mean()
        else:
            return paddle.nn.functional.mse_loss(
                input=model_pred.astype(dtype="float32"), label=target.astype(dtype="float32"), reduction="mean"
            )

    def process_batch(batch: dict):
        nonlocal global_step
        nonlocal next_save_iter
        now = time.time()
        with accelerator.accumulate(unet):
            logging_data = {}
            if global_step == 0:
                if accelerator.is_main_process and args.run_validation_at_start:
                    run_validation(step=global_step, node_index=accelerator.process_index // 8)
                accelerator.wait_for_everyone()
            loss = compute_loss_from_batch(batch)
            accelerator.backward(grad_tensor=loss, loss=loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(temporal_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
        fll = round(global_step * 100 / args.max_train_steps)
        fll = round(fll / 4)
        pr = bar(fll)  # noqa: *
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr(), "loss_time": time.time() - now}
        if (
            args.validate_every_steps is not None
            and global_step > min_steps_before_validation
            and global_step % args.validate_every_steps == 0
        ):
            if accelerator.is_main_process:
                run_validation(step=global_step, node_index=accelerator.process_index // 8)
            accelerator.wait_for_everyone()
        for key, val in logging_data.items():
            logs[key] = val
        progress_bar.set_postfix(**logs)
        progress_bar.set_description_str("Progress:" + pr)
        accelerator.log(logs, step=global_step)
        if (
            accelerator.is_main_process
            and next_save_iter is not None
            and global_step < args.max_train_steps
            and global_step + 1 == next_save_iter
        ):
            save_checkpoint()
            paddle.device.cuda.empty_cache()
            gc.collect()
            next_save_iter += args.save_n_steps

    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(dataloader):
            process_batch(batch)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            logger.info("Max train steps reached. Breaking while loop")
            break
        accelerator.wait_for_everyone()
    save_model_and_wait()
    accelerator.end_training()


if __name__ == "__main__":
    # paddle.multiprocessing.set_start_method('spawn')
    main()
