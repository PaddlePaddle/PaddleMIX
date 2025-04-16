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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # use pretrained vae kl-8.ckpt (CompVis/stable-diffusion-v1-4/vae)
    model_id: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "pretrained_vae_name_or_path"},
    )
    log_path: Optional[str] = field(default="./log_sd", metadata={"help": "unet_config_file"})
    ckpt_only_path: Optional[str] = field(default=None, metadata={"help": "unet_config_file"})
    train_iters: Optional[int] = field(
        default=1000000,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    log_iters: Optional[int] = field(default=100, metadata={"help": "Pretrained tokenizer model_max_length"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "num_inference_steps"})
    resolution: int = field(
        default=32,
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    lr: Optional[float] = field(default=1e-5, metadata={"help": "Log image every X steps."})
    initialie_generator: bool = field(default=False, metadata={"help": "enable_xformers_memory_efficient_attention."})
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "prediction_type, prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)"
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    wandb_iters: Optional[int] = field(
        default=100,
        metadata={"help": "profiler_options."},
    )
    # max_grad_norm: Optional[float] = field(
    #     default=10.0,
    #     metadata={"help": "profiler_options."},
    # )
    warmup_step: Optional[int] = field(
        default=500,
        metadata={"help": "profiler_options."},
    )
    min_step_percent: Optional[float] = field(
        default=0.02,
        metadata={"help": "profiler_options."},
    )
    max_step_percent: Optional[float] = field(
        default=0.98,
        metadata={"help": "profiler_options."},
    )
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=1,
    #     metadata={"help": "profiler_options."},
    # )
    num_train_timesteps: Optional[int] = field(
        default=1000,
        metadata={"help": "profiler_options."},
    )
    latent_resolution: Optional[int] = field(
        default=64,
        metadata={"help": "profiler_options."},
    )
    real_guidance_scale: Optional[float] = field(
        default=6.0,
        metadata={"help": "profiler_options."},
    )
    fake_guidance_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "profiler_options."},
    )
    grid_size: Optional[int] = field(
        default=2,
        metadata={"help": "profiler_options."},
    )
    no_save: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    cache_dir: Optional[str] = field(
        default="./.cache",
        metadata={"help": "profiler_options."},
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "profiler_options."},
    )
    latent_channel: Optional[int] = field(
        default=4,
        metadata={"help": "profiler_options."},
    )
    max_checkpoint: Optional[int] = field(
        default=150,
        metadata={"help": "profiler_options."},
    )
    dfake_gen_update_ratio: Optional[int] = field(
        default=1,
        metadata={"help": "profiler_options."},
    )
    generator_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": "profiler_options."},
    )
    guidance_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": "profiler_options."},
    )
    cls_on_clean_image: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    gen_cls_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    gen_cls_loss_weight: Optional[float] = field(
        default=0,
        metadata={"help": "profiler_options."},
    )
    guidance_cls_loss_weight: Optional[float] = field(
        default=0,
        metadata={"help": "profiler_options."},
    )
    sdxl: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    generator_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    conditioning_timestep: Optional[int] = field(
        default=999,
        metadata={"help": "profiler_options."},
    )
    tiny_vae: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    conditioning_timestep: Optional[int] = field(
        default=999,
        metadata={"help": "profiler_options."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    dm_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "profiler_options."},
    )
    denoising: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    denoising_timestep: Optional[int] = field(
        default=1000,
        metadata={"help": "profiler_options."},
    )
    num_denoising_step: Optional[int] = field(
        default=1,
        metadata={"help": "profiler_options."},
    )
    denoising_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "profiler_options."},
    )
    num_denoising_step: Optional[int] = field(
        default=1,
        metadata={"help": "profiler_options."},
    )
    diffusion_gan: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    diffusion_gan_max_timestep: Optional[int] = field(
        default=0,
        metadata={"help": "profiler_options."},
    )
    revision: str = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    real_image_path: str = field(
        default="",
        metadata={"help": "profiler_options."},
    )
    gan_alone: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    backward_simulation: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    generator_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    lora_rank: Optional[int] = field(
        default=64,
        metadata={"help": "profiler_options."},
    )
    lora_alpha: Optional[float] = field(
        default=8,
        metadata={"help": "profiler_options."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "profiler_options."},
    )
    use_fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
    train_prompt_path: Optional[str] = field(
        default="",
        metadata={"help": "profiler_options."},
    )
    log_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "profiler_options."},
    )
