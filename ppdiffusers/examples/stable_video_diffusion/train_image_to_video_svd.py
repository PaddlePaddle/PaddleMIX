# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import copy
import gc
import logging
import math
import os
import random
import shutil

# from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import paddle
import PIL
from einops import rearrange

# from huggingface_hub import create_repo, upload_folder
from paddle.io import Dataset, RandomSampler
from PIL import Image
from tqdm.auto import tqdm

import ppdiffusers
from ppdiffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)

# import paddle.nn.functional as F
# import paddle.distributed.fleet.utils.recompute
from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.logging import get_logger
from ppdiffusers.accelerate.utils import ProjectConfiguration, set_seed
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.training_utils import EMAModel
from ppdiffusers.transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from ppdiffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    load_image,
)
from ppdiffusers.utils.import_utils import is_ppxformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = paddle.arange(group, n, groups, dtype=dtype)
    u = paddle.rand(shape, dtype=dtype)
    return (offsets + u) / n


def rand_cosine_interpolated(
    shape,
    image_d,
    noise_d_low,
    noise_d_high,
    sigma_data=1.0,
    min_value=1e-3,
    max_value=1e3,
    dtype=paddle.float32,
):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * paddle.log(paddle.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return paddle.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return paddle.exp(-logsnr / 2) * sigma_data


def rand_log_normal(shape, loc=0.0, scale=1.0, dtype=paddle.float32):
    """Draws samples from a lognormal distribution without using icdf."""
    # westfish: paddle do not have icdf, so we use normal distribution to generate lognormal
    # u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    # return torch.distributions.Normal(loc, scale).icdf(u).exp()
    normal_samples = paddle.normal(mean=loc, std=scale, shape=shape)
    log_normal_samples = paddle.exp(normal_samples)
    return log_normal_samples


# min_value = 0.002
# max_value = 700
# image_d = 64
# noise_d_low = 32
# noise_d_high = 64
# sigma_data = 0.5

# westfish
class DataLoader(paddle.io.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
    ):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        if sampler is not None:
            self.batch_sampler.sampler = sampler


class DummyDataset(Dataset):
    def __init__(
        self, num_samples=100000, width=1024, height=576, sample_frames=25, train_data_dir="bdd100k/images/track/mini"
    ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        self.base_folder = train_data_dir
        self.folders = os.listdir(self.base_folder)
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)
        frames = os.listdir(folder_path)
        # Sort the frames by name
        frames.sort()

        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames."
            )

        # Randomly select a start index for frame sequence
        start_idx = random.randint(0, len(frames) - self.sample_frames)
        selected_frames = frames[start_idx : start_idx + self.sample_frames]

        # Initialize a tensor to store the pixel values
        pixel_values = paddle.empty((self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = paddle.to_tensor(np.array(img_resized)).astype(dtype="float32")

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1

                # Rearrange channels if necessary
                if self.channels == 3:
                    img_normalized = img_normalized.transpose(perm=[2, 0, 1])  # For RGB images
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(axis=2, keepdim=True)  # For grayscale images

                pixel_values[i] = img_normalized
        return {"pixel_values": pixel_values}


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = paddle.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].astype(input.dtype)

    tmp_kernel = tmp_kernel.expand([-1, c, -1, -1])

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = paddle.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape([-1, 1, height, width])
    input = input.reshape([-1, tmp_kernel.shape[0], input.shape[-2], input.shape[-1]])

    # convolve the tensor with the kernel.
    output = paddle.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.shape[0], padding=0, stride=1)

    out = output.reshape([b, c, h, w])
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = paddle.to_tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (paddle.arange(end=window_size, dtype=sigma.dtype) - window_size // 2).expand([batch_size, -1])

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = paddle.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = paddle.to_tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.astype(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].reshape([bs, 1]))
    kernel_y = _gaussian(ky, sigma[:, 0].reshape([bs, 1]))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train Stable Diffusion XL for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
        default=1e-4,
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
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    # parser.add_argument(
    #     "--allow_tf32",
    #     action="store_true",
    #     help=(
    #         "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
    #         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    #     ),
    # )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument(
    #     "--push_to_hub",
    #     action="store_true",
    #     help="Whether or not to push the model to the Hub.",
    # )
    # parser.add_argument(
    #     "--hub_token",
    #     type=str,
    #     default=None,
    #     help="The token to use to push to the Model Hub.",
    # )
    # parser.add_argument(
    #     "--hub_model_id",
    #     type=str,
    #     default=None,
    #     help="The name of the repository to keep in sync with the local `output_dir`.",
    # )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    # westfish
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="train_data",
        help=("The directory containing the training data. "),
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        default="demo.jpg",
        help=("The directory containing the validation data. "),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = paddle.Generator().manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb  # noqa: F401

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        ppdiffusers.utils.logging.set_verbosity_info()
    else:
        ppdiffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(  # noqa: F841
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16"
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    # Freeze vae and image_encoder
    def set_requires_grad(model, is_enabled):
        for param in model.parameters():
            param.stop_gradient = not is_enabled

    set_requires_grad(vae, False)
    set_requires_grad(image_encoder, False)
    set_requires_grad(unet, False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = paddle.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = paddle.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = paddle.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_ppxformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if True:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        # accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # if args.allow_tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        raise Exception("8-bit Adam not supported yet. Please use the default AdamW optimizer instead.")
    else:
        optimizer_cls = paddle.optimizer.AdamW

    def set_requires_grad(model, is_enabled):
        for param in model.parameters():
            param.stop_gradient = not is_enabled

    set_requires_grad(unet, True)
    parameters_list = []

    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.

    for name, para in unet.named_parameters():
        if "temporal_transformer_block" in name:
            parameters_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False

    optimizer = optimizer_cls(
        parameters=parameters_list,
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
    )

    # optimizer = optimizer_cls(
    #     unet.parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open("rec_para.txt", "w")
        rec_txt2 = open("rec_para_train.txt", "w")
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f"{name}\n")
            else:
                rec_txt2.write(f"{name}\n")
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = DummyDataset(
        width=args.width, height=args.height, sample_frames=args.num_frames, train_data_dir=args.train_data_dir
    )
    sampler = RandomSampler(train_dataset)
    # westfish: add sampler to self defined dataloader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    # westfish: add lr_scheduler to optimizer
    optimizer.set_lr_scheduler(lr_scheduler)

    # # Prepare everything with our `accelerator`.
    # unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
    #     unet, optimizer, lr_scheduler, train_dataloader
    # )
    # if args.use_ema:
    #     ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pd",
        ).pixel_values

        pixel_values = pixel_values.astype(weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        # passed_add_embed_dim = unet.module.config.addition_time_embed_dim * len(add_time_ids)
        # expected_add_embed_dim = unet.module.add_embedding.linear_1.weight.shape[0]
        passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.weight.shape[0]

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = paddle.to_tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.tile([batch_size, 1])
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # westfish: add amp
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    unet = paddle.amp.decorate(models=unet.to(dtype=paddle.float32), level="O2")
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # first, convert images to latent space.
                pixel_values = batch["pixel_values"].astype(weight_dtype)
                conditional_pixel_values = pixel_values[:, 0:1, :, :, :]

                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise that we'll add to the latents
                noise = paddle.randn(latents.shape, dtype=latents.dtype)
                bsz = latents.shape[0]

                cond_sigmas = rand_log_normal(
                    shape=[
                        bsz,
                    ],
                    loc=-3.0,
                    scale=0.5,
                ).astype(latents.dtype)
                noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = (
                    paddle.randn(conditional_pixel_values.shape, dtype=conditional_pixel_values.dtype) * cond_sigmas
                    + conditional_pixel_values
                )
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(
                    shape=[
                        bsz,
                    ],
                    loc=0.7,
                    scale=1.6,
                )
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                # westfish: paddle.Tensor do not support list input
                timesteps = paddle.Tensor(np.array([0.25 * sigma.log().squeeze() for sigma in sigmas]))

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :].astype(dtype="float32"))

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7,  # fixed
                    127,  # motion_bucket_id = 127, fixed
                    noise_aug_strength,  # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = paddle.rand([bsz], generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape([bsz, 1, 1])
                    # Final text conditioning.
                    null_conditioning = paddle.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = paddle.where(
                        prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1)
                    )
                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).astype(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).astype(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape([bsz, 1, 1, 1])
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(1).tile([1, noisy_latents.shape[1], 1, 1, 1])
                inp_noisy_latents = paddle.concat(
                    [inp_noisy_latents, conditional_latents.astype(inp_noisy_latents.dtype)], axis=2
                )

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents

                # westfish: add amp
                with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level="O2"):
                    model_pred = unet(
                        inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids
                    ).sample

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas**2) * (sigmas**-2.0)

                # MSE loss
                loss = paddle.mean(
                    (
                        weighing.astype("float32")
                        * (denoised_latents.astype("float32") - target.astype("float32")) ** 2
                    ).reshape([target.shape[0], -1]),
                    axis=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.tile([args.per_gpu_batch_size])).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                # accelerator.backward(loss)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
                scaler.update()
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # westfish: save_state substitute
                        # accelerator.save_state(save_path)
                        models = []
                        models.append(copy.deepcopy(unet))
                        weights = []

                        def get_state_dict(model):
                            state_dict = model.state_dict()
                            if state_dict is not None:
                                for k in state_dict:
                                    if getattr(state_dict[k], "dtype", None) == paddle.float16:
                                        state_dict[k] = state_dict[k]._to(dtype="float32")
                            return state_dict

                        for i, model in enumerate(models):
                            weights.append(get_state_dict(model))
                        save_model_hook(models, weights, save_path)
                        del models, weights
                        gc.collect()
                        paddle.device.cuda.empty_cache()

                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if (global_step % args.validation_steps == 0) or (global_step == 1):
                        logger.info(f"Running validation... \n Generating {args.num_validation_images} videos.")
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = StableVideoDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        # with paddle.amp.auto_cast(
                        #     enable=True, custom_white_list=None, custom_black_list=None, level="O2"
                        # ):
                        if True:
                            for val_img_idx in range(args.num_validation_images):
                                num_frames = args.num_frames
                                video_frames = pipeline(
                                    load_image(args.valid_data_path).resize((args.width, args.height)),
                                    height=args.height,
                                    width=args.width,
                                    num_frames=num_frames,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127,
                                    fps=7,
                                    noise_aug_strength=0.02,
                                    # generator=generator,
                                ).frames[0]

                                out_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.mp4",
                                )

                                for i in range(num_frames):
                                    img = video_frames[i]
                                    video_frames[i] = np.array(img)
                                export_to_gif(video_frames, out_file, 8)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        paddle.device.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_lr()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        # if args.push_to_hub:
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )
    accelerator.end_training()


if __name__ == "__main__":
    main()
