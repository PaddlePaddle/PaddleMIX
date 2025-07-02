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

# code is heavily based on https://github.com/tianweiy/DMD2

import argparse
import logging
import time

import numpy as np
import paddle
import wandb
from coco_eval.coco_evaluator import evaluate_model
from safetensors.paddle import load_file
from sdxl.sdxl_text_encoder import SDXLTextEncoder
from tqdm import tqdm
from utils import SDTextDataset, create_image_grid

from ppdiffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.logging import get_logger
from ppdiffusers.accelerate.utils import ProjectConfiguration, set_seed
from ppdiffusers.peft import LoraConfig
from ppdiffusers.transformers import AutoTokenizer

logger = get_logger(__name__, log_level="INFO")


def create_generator(checkpoint_path, base_model=None, args=None):
    if base_model is None:
        generator = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
        ).float()
        generator.requires_grad_(False)

        if args.generator_lora:
            lora_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ]
            lora_config = LoraConfig(
                r=args.lora_rank,
                target_modules=lora_target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            generator.add_adapter(lora_config)
    else:
        generator = base_model

    # sometime the state_dict is not fully saved yet
    counter = 0
    while True:
        try:
            if checkpoint_path.endswith("safetensors"):
                state_dict = load_file(checkpoint_path)
            else:
                state_dict = paddle.load(checkpoint_path)
            break
        except Exception as e:
            print(f"fail to load checkpoint {checkpoint_path}", e)
            time.sleep(1)

            counter += 1

            if counter > 100:
                return None

    print(generator.set_state_dict(state_dict))

    return generator


def build_condition_input(resolution, accelerator):
    original_size = (resolution, resolution)
    target_size = (resolution, resolution)
    crop_top_left = (0, 0)

    add_time_ids = list(original_size + crop_top_left + target_size)
    add_time_ids = paddle.to_tensor([add_time_ids], dtype=paddle.float32)
    return add_time_ids


def get_x0_from_noise(sample, model_output, timestep, alphas_cumprod):
    alpha_prod_t = alphas_cumprod[timestep].reshape([-1, 1, 1, 1])
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


@paddle.no_grad()
def sample(
    noise,
    unet_added_conditions,
    model,
    vae,
    noise_scheduler,
    prompt_embed,
    device="cuda",
    num_step=1,
    conditioning_timestep=999,
):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    if num_step == 1:
        all_timesteps = [conditioning_timestep]
        step_interval = 0
    elif num_step == 2:
        all_timesteps = [999, 499]
        step_interval = 500
    elif num_step == 3:
        raise NotImplementedError()
        all_timesteps = [999, 749, 499]
        step_interval = 250
    elif num_step == 4:
        all_timesteps = [999, 749, 499, 249]
        step_interval = 250
    else:
        raise NotImplementedError()

    DTYPE = prompt_embed.dtype

    for constant in all_timesteps:
        current_timesteps = paddle.ones(len(prompt_embed), dtype=paddle.int64) * constant
        eval_images = model(noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions).sample

        eval_images = get_x0_from_noise(noise, eval_images, current_timesteps, alphas_cumprod).cast(paddle.float32)

        next_timestep = current_timesteps - step_interval
        noise = noise_scheduler.add_noise(eval_images, paddle.randn_like(eval_images), next_timestep).to(dtype=DTYPE)

    eval_images = vae.decode(eval_images / vae.config.scaling_factor, return_dict=False)[0]
    eval_images = ((eval_images + 1.0) * 127.5).clip(0, 255).to(paddle.uint8).permute(0, 2, 3, 1)
    return eval_images


@paddle.no_grad()
def evaluate():
    paddle.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="pass to folder list")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--image_resolution", type=int, default=1024)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--eval_res", type=int, default=256)
    parser.add_argument("--ref_dir", type=str)
    parser.add_argument("--total_eval_samples", type=int, default=30000)
    parser.add_argument("--per_image_object", type=int, default=9)
    parser.add_argument("--test_visual_batch_size", type=int, default=81)
    parser.add_argument("--predict_x0", action="store_true")
    parser.add_argument("--anno_path", type=str)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str)
    parser.add_argument("--sdxl_lightning_4step", action="store_true")
    parser.add_argument("--sdxl_lightning_1step", action="store_true")
    parser.add_argument("--clip_score", action="store_true")
    parser.add_argument("--sdxl_teacher", action="store_true")
    parser.add_argument("--num_step", type=int, default=1)
    parser.add_argument("--lcm_1step", action="store_true")
    parser.add_argument("--lcm_4step", action="store_true")
    parser.add_argument("--turbo_1step", action="store_true")
    parser.add_argument("--turbo_4step", action="store_true")
    parser.add_argument("--image_reward", action="store_true")
    parser.add_argument("--pick_score", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, help="specify a single checkpoint instead of a folder")
    parser.add_argument("--guidance_scale", type=float, default=6)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()

    folder = args.folder

    accelerator_project_config = ProjectConfiguration(logging_dir=args.folder)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # paddle.backends.cuda.matmul.allow_tf32 = True
    # paddle.backends.cudnn.allow_tf32 = True

    if accelerator.is_main_process:
        run = wandb.init(
            config=args,
            dir=args.folder,
            **{"mode": "offline", "entity": args.wandb_entity, "project": args.wandb_project},
        )
        wandb.run.name = args.wandb_name
        print(run.dir)

    evaluated_checkpoints = set()

    text_encoder = SDXLTextEncoder(args, accelerator)
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    tokenizer_two = AutoTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    dataset = SDTextDataset(
        anno_path=args.anno_path, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two, is_sdxl=True
    )

    dataloader = paddle.io.DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, num_workers=8
    )
    base_add_time_ids = build_condition_input(args.image_resolution, accelerator)

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae").to(
        dtype=paddle.float32
    )

    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

    generator = None

    if args.sdxl_teacher:
        teacher_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float32, use_safetensors=True
        )
        # teacher_pipeline = teacher_pipeline.to(accelerator.device)

        teacher_pipeline.set_progress_bar_config(disable=True)
        teacher_pipeline.safety_checker = None
    elif args.sdxl_lightning_4step:
        base = "stabilityai/stable-diffusion-xl-base-1.0"

        # Load model.
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(accelerator.device, paddle.float32)
        # unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
        lightning_pipeline = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, paddle_dtype=paddle.float32
        ).to(accelerator.device)

        # Ensure sampler uses "trailing" timesteps.
        lightning_pipeline.scheduler = EulerDiscreteScheduler.from_config(
            lightning_pipeline.scheduler.config, timestep_spacing="trailing"
        )
        lightning_pipeline.safety_checker = None
        lightning_pipeline.set_progress_bar_config(disable=True)

    elif args.sdxl_lightning_1step:
        base = "stabilityai/stable-diffusion-xl-base-1.0"

        # Load model.
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(accelerator.device, paddle.float32)

        lightning_pipeline = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, paddle_dtype=paddle.float32)

        # Ensure sampler uses "trailing" timesteps.
        lightning_pipeline.scheduler = EulerDiscreteScheduler.from_config(
            lightning_pipeline.scheduler.config, timestep_spacing="trailing", prediction_type="sample"
        )
        lightning_pipeline.safety_checker = None
        lightning_pipeline.set_progress_bar_config(disable=True)

    elif args.lcm_1step or args.lcm_4step:
        unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", paddle_dtype=paddle.float32)
        lcm_pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, paddle_dtype=paddle.float32
        )
        lcm_pipeline.scheduler = LCMScheduler.from_config(lcm_pipeline.scheduler.config)
        lcm_pipeline.to(accelerator.device)

        lcm_pipeline.safety_checker = None
        lcm_pipeline.set_progress_bar_config(disable=True)

    elif args.turbo_1step or args.turbo_4step:
        turbo_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", paddle_dtype=paddle.float32)
        turbo_pipe.to(accelerator.device)

        turbo_pipe.safety_checker = None
        turbo_pipe.set_progress_bar_config(disable=True)

    # while True:
    # if args.checkpoint_path:
    #     new_checkpoints = [args.checkpoint_path]
    # else:
    #     new_checkpoints = sorted(glob.glob(os.path.join(folder, "*checkpoint_model_*")))

    #     new_checkpoints = set(new_checkpoints) - evaluated_checkpoints
    #     new_checkpoints = sorted(list(new_checkpoints))

    # print('new_checkpoints', new_checkpoints)
    # if len(new_checkpoints) == 0:
    #     continue
    new_checkpoints = [args.checkpoint_path]

    for checkpoint in new_checkpoints:
        print(f"Evaluating {folder} {checkpoint}")
        try:
            model_index = int(checkpoint.replace("/", "").split("_")[-1])
        except Exception as e:
            print(e)
            model_index = 0

        generator = create_generator(
            # os.path.join(checkpoint, "model.safetensors"),
            checkpoint,
            base_model=generator,
            args=args,
        )

        if generator is None:
            continue

        generator = generator.to(accelerator.device)

        set_seed(args.seed + accelerator.process_index)

        all_images = []
        all_captions = []

        for i, data in tqdm(
            enumerate(dataloader), disable=not accelerator.is_main_process, total=len(dataset) // args.eval_batch_size
        ):
            all_captions.append(data["key"])

            prompt_embed, pooled_prompt_embed = text_encoder(data)

            noise = paddle.randn(
                [len(prompt_embed), 4, args.latent_resolution, args.latent_resolution],
                dtype=paddle.float32,
                generator=paddle.Generator().manual_seed(i),
            ).to(accelerator.device)
            # noise = paddle.load('/root/paddlejob/workspace/env_run/jll/DMD2/noise_sdxl_infer.pkl')
            add_time_ids = base_add_time_ids.tile([noise.shape[0], 1])

            unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embed}

            if args.sdxl_teacher:
                eval_images = teacher_pipeline(
                    prompt_embeds=prompt_embed,
                    pooled_prompt_embeds=pooled_prompt_embed,
                    latents=noise,
                    guidance_scale=args.guidance_scale,
                    output_type="np",
                ).images
                eval_images = (paddle.to_tensor(eval_images, dtype=paddle.float32) * 255.0).to(paddle.uint8)
            elif args.sdxl_lightning_4step:
                eval_images = lightning_pipeline(
                    prompt_embeds=prompt_embed,
                    pooled_prompt_embeds=pooled_prompt_embed,
                    latents=noise,
                    num_inference_steps=4,
                    guidance_scale=0,
                    output_type="np",
                ).images
                eval_images = (paddle.to_tensor(eval_images, dtype=paddle.float32) * 255.0).to(paddle.uint8)
            elif args.sdxl_lightning_1step:
                eval_images = lightning_pipeline(
                    prompt_embeds=prompt_embed,
                    pooled_prompt_embeds=pooled_prompt_embed,
                    latents=noise,
                    num_inference_steps=1,
                    guidance_scale=0,
                    output_type="np",
                ).images
                eval_images = (paddle.to_tensor(eval_images, dtype=paddle.float32) * 255.0).to(paddle.uint8)
            elif args.lcm_1step or args.lcm_4step:
                eval_images = lcm_pipeline(
                    prompt_embeds=prompt_embed,
                    pooled_prompt_embeds=pooled_prompt_embed,
                    latents=noise,
                    num_inference_steps=1 if args.lcm_1step else 4,
                    guidance_scale=0,
                    output_type="np",
                ).images
                eval_images = (paddle.to_tensor(eval_images, dtype=paddle.float32) * 255.0).to(paddle.uint8)
            elif args.turbo_1step or args.turbo_4step:
                eval_images = turbo_pipe(
                    prompt_embeds=prompt_embed,
                    pooled_prompt_embeds=pooled_prompt_embed,
                    latents=noise,
                    num_inference_steps=1 if args.turbo_1step else 4,
                    guidance_scale=0,
                    output_type="np",
                ).images
                eval_images = (paddle.to_tensor(eval_images, dtype=paddle.float32) * 255.0).to(paddle.uint8)
            else:
                eval_images = sample(
                    noise,
                    unet_added_conditions,
                    generator,
                    vae,
                    scheduler,
                    prompt_embed,
                    device=accelerator.device,
                    num_step=args.num_step,
                    conditioning_timestep=args.conditioning_timestep,
                )
            all_images.append(eval_images.cpu().numpy())

        all_images = np.concatenate(all_images, axis=0)[: args.total_eval_samples]
        all_captions = [caption for sublist in all_captions for caption in sublist]

        data_dict = {"all_images": all_images, "all_captions": all_captions}

        if args.result_path is not None:
            paddle.save(data_dict, args.result_path, pickle_protocol=5)

        paddle.device.cuda.empty_cache()

        if accelerator.is_main_process:
            fid = evaluate_model(args, accelerator.device, data_dict["all_images"], patch_fid=False)
            print("fid", fid)

            if args.clip_score:
                raise NotImplementedError("not support clip score yet")

            if args.image_reward:
                raise NotImplementedError("not support image reward yet")

            visualize_images = all_images[: args.test_visual_batch_size]

            image_brightness = (visualize_images / 255.0).mean()
            image_std = (visualize_images / 255.0).std()

            wandb.log({"image_brightness": image_brightness, "image_std": image_std}, step=model_index)

            for start in range(0, len(visualize_images), args.per_image_object):
                end = min(start + args.per_image_object, len(visualize_images))

                if start >= end:
                    continue

                eval_images_grid = create_image_grid(args, visualize_images[start:end], None)

                wandb.log(
                    {f"generated_image_grid_{start:04d}_{end:04d}": wandb.Image(eval_images_grid)}, step=model_index
                )

        evaluated_checkpoints.add(checkpoint)


if __name__ == "__main__":
    evaluate()
