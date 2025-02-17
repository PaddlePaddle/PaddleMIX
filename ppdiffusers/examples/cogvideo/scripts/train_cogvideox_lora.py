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
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import AutoTokenizer
from tqdm.auto import tqdm

import ppdiffusers
from ppdiffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.logging import get_logger
from ppdiffusers.accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from ppdiffusers.image_processor import VaeImageProcessor
from ppdiffusers.models.embeddings import get_3d_rotary_pos_embed
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from ppdiffusers.pipelines.cogvideo.pipeline_cogvideox import (
    get_resize_crop_region_for_grid,
)
from ppdiffusers.transformers import T5EncoderModel, T5Tokenizer
from ppdiffusers.utils import (
    convert_unet_state_dict_to_peft,
    export_to_video_2,
    is_wandb_available,
)

if ppdiffusers.utils.is_wandb_available():
    import wandb

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument("--instance_data_root", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help="Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6, help="The guidance scale to use while sampling validation videos."
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--rank", type=int, default=128, help="The dimension of the LoRA update matrices.")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help="The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        choices=["O0", "O1", "O2"],
        help="Level of automatic mixed precision",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--height", type=int, default=480, help="All input videos are resized to this height.")
    parser.add_argument("--width", type=int, default=720, help="All input videos are resized to this width.")
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default="center",
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument("--random_flip", action="store_true", help="whether to randomly flip videos horizontally")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.",
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.',
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
        default=0.0001,
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
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help="The optimizer type to use.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0001, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory where logs are stored.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.',
    )
    return parser.parse_args()


class VideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        video_reshape_mode: str = "center",
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found len(self.instance_prompts)={len(self.instance_prompts)!r} and len(self.instance_video_paths)={len(self.instance_video_paths)!r}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        self.instance_videos = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video": self.instance_videos[index],
        }

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")
        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)
        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )
        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]
        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def resize(self, img, size, interpolation="bilinear", data_format="CHW"):
        """
        Resizes the image to given size

        Args:
            input (paddle.Tensor): Image to be resized.
            size (int|list|tuple): Target size of input data, with (height, width) shape.
            interpolation (int|str, optional): Interpolation method. when use paddle backend,
                support method are as following:
                - "nearest"
                - "bilinear"
                - "bicubic"
                - "trilinear"
                - "area"
                - "linear"
            data_format (str, optional): paddle.Tensor format
                - 'CHW'
                - 'HWC'
        Returns:
            paddle.Tensor: Resized image.

        """

        if not (isinstance(size, int) or (isinstance(size, (tuple, list)) and len(size) == 2)):
            raise TypeError(f"Got inappropriate size arg: {size}")

        def _get_image_w_axis(data_format):
            if data_format.lower() == "chw":
                return -1
            elif data_format.lower() == "hwc":
                return -2

        def _get_image_h_axis(data_format):
            if data_format.lower() == "chw":
                return -2
            elif data_format.lower() == "hwc":
                return -3

        def _get_image_size(img, data_format):
            return (
                img.shape[_get_image_w_axis(data_format)],
                img.shape[_get_image_h_axis(data_format)],
            )

        if isinstance(size, int):
            w, h = _get_image_size(img, data_format)
            # TODO(Aurelius84): In static graph mode, w and h will be -1 for dynamic shape.
            # We should consider to support this case in future.
            if w <= 0 or h <= 0:
                raise NotImplementedError(f"Not support while w<=0 or h<=0, but received w={w}, h={h}")
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
        else:
            oh, ow = size

        img = F.interpolate(
            img,
            size=(oh, ow),
            mode=interpolation.lower(),
            data_format="N" + data_format.upper(),
        )

        return img

    def _resize_for_rectangle_crop(self, arr):

        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode
        if tuple(arr.shape)[3] / tuple(arr.shape)[2] > image_size[1] / image_size[0]:
            arr = self.resize(
                img=arr,
                size=[image_size[0], int(tuple(arr.shape)[3] * image_size[0] / tuple(arr.shape)[2])],
                interpolation="bicubic",
            )
        else:
            arr = self.resize(
                img=arr,
                size=[int(tuple(arr.shape)[2] * image_size[1] / tuple(arr.shape)[3]), image_size[1]],
                interpolation="bicubic",
            )
        h, w = tuple(arr.shape)[2], tuple(arr.shape)[3]
        arr = arr.squeeze(axis=0)
        delta_h = h - image_size[0]
        delta_w = w - image_size[1]
        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = paddle.vision.transforms.crop(img=arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        progress_dataset_bar = tqdm(
            range(0, len(self.instance_video_paths)), desc="Loading progress resize and crop videos"
        )
        videos = []
        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(uri=filename.as_posix())
            video_num_frames = len(video_reader)
            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame]).asnumpy()
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame))).asnumpy()
            else:
                indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices).asnumpy()

            frames = frames[: self.max_num_frames]
            selected_num_frames = tuple(frames.shape)[0]
            remainder = (3 + selected_num_frames % 4) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = tuple(frames.shape)[0]
            assert (selected_num_frames - 1) % 4 == 0
            frames = paddle.to_tensor(frames)
            frames = (frames - 127.5) / 127.5

            frames = frames.transpose(perm=[0, 3, 1, 2])

            progress_dataset_bar.set_description(
                f"Loading progress Resizing video from {tuple(frames.shape)[2]}x{tuple(frames.shape)[3]} to {self.height}x{self.width}"
            )
            frames = self._resize_for_rectangle_crop(frames)

            videos.append(frames.contiguous())
            progress_dataset_bar.update(1)
        progress_dataset_bar.close()

        return videos


def log_validation(pipe, args, accelerator, pipeline_args, epoch, is_final_validation: bool = False):
    logger.info(
        f"""Running validation... Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."""
    )
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)

    generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
    videos = []
    for _ in range(args.num_validation_videos):
        pd_images = pipe(**pipeline_args, generator=generator, output_type="pd").frames[0]
        pd_images = paddle.stack(x=[pd_images[i] for i in range(tuple(pd_images.shape)[0])])
        image_np = pd_images.cast("float32").transpose([0, 2, 3, 1]).cpu().numpy()
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        videos.append(image_pil)
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video_2(video, filename, fps=8)
                video_filenames.append(filename)
            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )
    del pipe

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    dtype: Optional[paddle.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids)[0]

    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = tuple(prompt_embeds.shape)
    prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_videos_per_prompt, 1])
    prompt_embeds = prompt_embeds.reshape([batch_size * num_videos_per_prompt, seq_len, -1])
    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    dtype: Optional[paddle.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
        )
    else:
        with paddle.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )
    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize):

    supported_optimizers = ["adam", "adamw"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"
    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        optimizer_class = paddle.optimizer.AdamW
        optimizer = optimizer_class(
            learning_rate=params_to_optimize[0]["lr"],
            parameters=params_to_optimize[0]["params"],
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = paddle.optimizer.Adam
        optimizer = optimizer_class(
            learning_rate=params_to_optimize[0]["lr"],
            parameters=params_to_optimize[0]["params"],
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    else:
        raise NotImplementedError

    return optimizer


def cast_training_params(model: Union[paddle.nn.Layer, List[paddle.nn.Layer]], dtype=paddle.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if not param.stop_gradient:
                param.set_value(param.to(dtype=dtype))


def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        fp16_opt_level=args.fp16_opt_level,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        ppdiffusers.utils.logging.set_verbosity_info()
    else:
        ppdiffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    load_dtype = paddle.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else paddle.float16

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", paddle_dtype=load_dtype, low_cpu_mem_usage=True
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        variant=args.variant,
        low_cpu_mem_usage=True,
        paddle_dtype=load_dtype,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.stop_gradient = True
    transformer.stop_gradient = True
    vae.stop_gradient = True

    weight_dtype = paddle.float32

    if args.mixed_precision == "fp16":
        weight_dtype = paddle.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = paddle.bfloat16

    text_encoder.to(dtype=weight_dtype)
    transformer.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model

    def save_model_hook(models, weights, output_dir):

        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                weights.pop()
            CogVideoXPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers_to_save)

    def load_model_hook(models, input_dir):

        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }

        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model:  {unexpected_keys}. "
                )

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=paddle.float32)

    args.learning_rate *= 0.01  # paddle learning_rate needs to be scaled by 0.01

    transformer_lora_parameters = list(filter(lambda p: not p.stop_gradient, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}

    params_to_optimize = [transformer_parameters_with_lr]
    optimizer = get_optimizer(args, params_to_optimize)

    train_dataset = VideoDataset(
        instance_data_root=args.instance_data_root,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column,
        video_column=args.video_column,
        height=args.height,
        width=args.width,
        video_reshape_mode=args.video_reshape_mode,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    def encode_video(video, bar):
        bar.update(1)
        video = video.to(dtype=vae.dtype).unsqueeze(axis=0)
        video = video.transpose(perm=[0, 2, 1, 3, 4])
        latent_dist = vae.encode(video).latent_dist

        return latent_dist

    progress_encode_bar = tqdm(range(0, len(train_dataset.instance_videos)), desc="Loading Encode videos")

    train_dataset.instance_videos = [
        encode_video(video, progress_encode_bar) for video in train_dataset.instance_videos
    ]
    progress_encode_bar.close()

    def collate_fn(examples):
        videos = [(example["instance_video"].sample() * vae.config.scaling_factor) for example in examples]
        prompts = [example["instance_prompt"] for example in examples]
        videos = paddle.concat(x=videos)
        videos = videos.transpose(perm=[0, 2, 1, 3, 4])
        videos = videos.astype(dtype="float32")
        return {"videos": videos, "prompts": prompts}

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.size for model in params_to_optimize for param in model["params"])
    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                model_input = batch["videos"].to(dtype=weight_dtype)
                prompts = batch["prompts"]

                prompt_embeds = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    prompts,
                    model_config.max_text_seq_length,
                    weight_dtype,
                    requires_grad=False,
                )

                noise = paddle.randn(shape=model_input.shape, dtype=model_input.dtype)

                batch_size, num_frames, num_channels, height, width = tuple(model_input.shape)
                timesteps = paddle.randint(low=0, high=scheduler.config.num_train_timesteps, shape=(batch_size,))
                timesteps = timesteps.astype(dtype="int64")

                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )
                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(tuple(weights.shape)) < len(tuple(model_pred.shape)):
                    weights = weights.unsqueeze(axis=-1)
                target = model_input
                loss = paddle.mean(x=(weights * (model_pred - target) ** 2).reshape([batch_size, -1]), axis=1)
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()

                    optimizer.zero_grad()
                lr_scheduler.step()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()}

            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:

                pipe = CogVideoXPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    text_encoder=unwrap_model(text_encoder),
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    paddle_dtype=weight_dtype,
                )

                validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                for validation_prompt in validation_prompts:
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                    }
                    validation_outputs = log_validation(
                        pipe=pipe, args=args, accelerator=accelerator, pipeline_args=pipeline_args, epoch=epoch
                    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            "float16"
            if args.mixed_precision == "fp16"
            else "bfloat16"
            if args.mixed_precision == "bf16"
            else "float32"
        )
        transformer = transformer.to(dtype=dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        CogVideoXPipeline.save_lora_weights(
            save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers
        )
        del transformer

        pipe = CogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, paddle_dtype=weight_dtype
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            for validation_prompt in validation_prompts:
                pipeline_args = {
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }
                video = log_validation(
                    pipe=pipe,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    is_final_validation=True,
                )
                validation_outputs.extend(video)

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
