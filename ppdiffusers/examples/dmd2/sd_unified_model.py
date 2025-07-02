# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# code is heavily based on https://github.com/tianweiy/DMD2

# A single unified model that wraps both the generator and discriminator
import paddle
from paddle import nn
from sd_guidance import SDGuidance
from sdxl.sdxl_text_encoder import SDXLTextEncoder
from utils import get_x0_from_noise

from ppdiffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
from ppdiffusers.accelerate.utils import broadcast
from ppdiffusers.peft import LoraConfig
from ppdiffusers.transformers import CLIPTextModel


class SDUniModel(nn.Layer):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        self.guidance_model = SDGuidance(args, accelerator)
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.num_visuals = args.grid_size * args.grid_size
        self.conditioning_timestep = args.conditioning_timestep
        self.use_fp16 = args.use_fp16
        self.gradient_checkpointing = args.gradient_checkpointing
        self.backward_simulation = args.backward_simulation

        self.cls_on_clean_image = args.cls_on_clean_image
        self.denoising = args.denoising
        self.denoising_timestep = args.denoising_timestep
        self.noise_scheduler = self.guidance_model.scheduler
        self.num_denoising_step = args.num_denoising_step
        self.denoising_step_list = paddle.to_tensor(
            list(range(self.denoising_timestep - 1, 0, -(self.denoising_timestep // self.num_denoising_step))),
            dtype=paddle.int64,
        )
        self.timestep_interval = self.denoising_timestep // self.num_denoising_step

        if args.initialie_generator:
            self.feedforward_model = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").float()

            if args.generator_lora:
                self.feedforward_model.requires_grad_(False)
                assert args.sdxl
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
                self.feedforward_model.add_adapter(lora_config)
            else:
                self.feedforward_model.requires_grad_(True)

            if self.gradient_checkpointing:
                self.feedforward_model.enable_gradient_checkpointing()
        else:
            raise NotImplementedError()

        self.sdxl = args.sdxl

        if self.sdxl:
            self.text_encoder = SDXLTextEncoder(args, accelerator).to(accelerator.device)
            self.text_encoder.requires_grad_(False)
            self.add_time_ids = self.build_condition_input(args.resolution, accelerator)
        else:
            self.text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(
                accelerator.device
            )
            self.text_encoder.requires_grad_(False)

        self.alphas_cumprod = self.guidance_model.alphas_cumprod.to(accelerator.device)

        self.not_sdxl_vae = not (self.sdxl and (not args.tiny_vae))

        if args.tiny_vae:
            if "stable-diffusion-xl" in args.model_id:
                self.vae = (
                    AutoencoderTiny.from_pretrained("madebyollin/taesdxl", paddle_dtype=paddle.float32)
                    .cast(paddle.float32)
                    .to(accelerator.device)
                )
            else:
                raise NotImplementedError()
        else:
            self.vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").float().to(accelerator.device)
        self.vae.requires_grad_(False)

        if self.use_fp16 and self.not_sdxl_vae:
            # "SDXL's origianl VAE doesn't work with half precision"
            self.vae.to(paddle.float16)

    def build_condition_input(self, resolution, accelerator):
        original_size = (resolution, resolution)
        target_size = (resolution, resolution)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = paddle.to_tensor([add_time_ids], dtype=paddle.float32)
        return add_time_ids

    def decode_image(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample.cast(paddle.float32)
        return image

    @paddle.no_grad()
    def sample_backward(self, noisy_image, real_text_embedding, real_pooled_text_embedding):
        batch_size = noisy_image.shape[0]

        add_time_ids = self.add_time_ids.tile([batch_size, 1])

        unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": real_pooled_text_embedding}

        # we choose a random step and share it across all gpu
        selected_step = paddle.randint(low=0, high=self.num_denoising_step, size=(1,), dtype=paddle.int64)
        selected_step = broadcast(selected_step, from_process=0)

        # set a default value in case we don't enter the loop
        # it will be overwriten in the pure_noise_mask check later
        generated_image = noisy_image

        for constant in self.denoising_step_list[:selected_step]:
            current_timesteps = paddle.ones(batch_size, dtype=paddle.int64) * constant
            generated_noise = self.feedforward_model(
                noisy_image, current_timesteps, real_text_embedding, added_cond_kwargs=unet_added_conditions
            ).sample

            generated_image = get_x0_from_noise(
                noisy_image,
                generated_noise.cast(paddle.float64),
                self.alphas_cumprod.cast(paddle.float64),
                current_timesteps,
            ).cast(paddle.float32)

            next_timestep = current_timesteps - self.timestep_interval
            noisy_image = self.noise_scheduler.add_noise(
                generated_image, paddle.randn_like(generated_image), next_timestep
            ).to(noisy_image.dtype)

        return_timesteps = self.denoising_step_list[selected_step] * paddle.ones(batch_size, dtype=paddle.int64)
        return generated_image, return_timesteps

    @paddle.no_grad()
    def prepare_denoising_data(self, denoising_dict, real_train_dict, noise):
        assert self.sdxl, "Denoising is only supported for SDXL"

        indices = paddle.randint(0, self.num_denoising_step, (noise.shape[0],), dtype=paddle.int64)
        timesteps = self.denoising_step_list[indices]
        # timesteps = self.denoising_step_list.to(noise.device)[indices]

        text_embedding, pooled_text_embedding = self.text_encoder(denoising_dict)

        if real_train_dict is not None:
            real_text_embedding, real_pooled_text_embedding = self.text_encoder(real_train_dict)

            real_train_dict["text_embedding"] = real_text_embedding

            real_unet_added_conditions = {
                "time_ids": self.add_time_ids.tile(len(real_text_embedding), 1),
                "text_embeds": real_pooled_text_embedding,
            }
            real_train_dict["unet_added_conditions"] = real_unet_added_conditions

        if self.backward_simulation:
            # we overwrite the denoising timesteps
            # note: we also use uncorrelated noise
            clean_images, timesteps = self.sample_backward(
                paddle.randn_like(noise), text_embedding, pooled_text_embedding
            )
        else:
            clean_images = denoising_dict["images"]  # .to(noise.device)

        noisy_image = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # set last timestep to pure noise
        pure_noise_mask = timesteps == (self.num_train_timesteps - 1)
        noisy_image[pure_noise_mask] = noise[pure_noise_mask]

        return timesteps, text_embedding, pooled_text_embedding, real_train_dict, noisy_image

    @paddle.no_grad()
    def prepare_pure_generation_data(self, text_embedding, real_train_dict, noise):

        # actually it is a tokenized prompt
        text_embedding_output = self.text_encoder(text_embedding)

        text_embedding = text_embedding_output[0].float()
        pooled_text_embedding = text_embedding_output[1].float()

        if real_train_dict is not None:
            if self.sdxl:
                real_text_embedding, real_pooled_text_embedding = self.text_encoder(real_train_dict)
                real_train_dict["text_embedding"] = real_text_embedding
                real_unet_added_conditions = {
                    "time_ids": self.add_time_ids.tile([len(real_train_dict["text_embedding"]), 1]),
                    "text_embeds": real_pooled_text_embedding,
                }
                real_train_dict["unet_added_conditions"] = real_unet_added_conditions
            else:
                real_text_embedding_output = self.text_encoder(real_train_dict["text_input_ids_one"].squeeze(1))
                real_train_dict["text_embedding"] = real_text_embedding_output[0].cast(paddle.float32)
                real_train_dict["unet_added_conditions"] = None

        noisy_image = noise
        return text_embedding, pooled_text_embedding, real_train_dict, noisy_image

    def forward(
        self,
        noise,
        text_embedding,
        uncond_embedding,
        visual=False,
        denoising_dict=None,
        real_train_dict=None,
        compute_generator_gradient=True,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None,
    ):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn)

        if generator_turn:
            if self.denoising:
                # we ignore the text_embedding, uncond_embedding passed to the model
                (
                    timesteps,
                    text_embedding,
                    pooled_text_embedding,
                    real_train_dict,
                    noisy_image,
                ) = self.prepare_denoising_data(denoising_dict, real_train_dict, noise)
            else:
                timesteps = paddle.ones(noise.shape[0], dtype=paddle.int64) * self.conditioning_timestep
                (
                    text_embedding,
                    pooled_text_embedding,
                    real_train_dict,
                    noisy_image,
                ) = self.prepare_pure_generation_data(text_embedding, real_train_dict, noise)

            if self.sdxl:
                add_time_ids = self.add_time_ids.tile([noise.shape[0], 1])
                unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_text_embedding}

                uncond_unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": paddle.zeros_like(pooled_text_embedding),
                }
                uncond_embedding = paddle.zeros_like(text_embedding)
            else:
                unet_added_conditions = None
                uncond_unet_added_conditions = None

            if compute_generator_gradient:
                if self.use_fp16:
                    with paddle.amp.auto_cast(dtype="bfloat16"):
                        generated_noise = self.feedforward_model(
                            noisy_image,
                            timesteps.cast(paddle.float64),
                            text_embedding,
                            added_cond_kwargs=unet_added_conditions,
                        ).sample
                else:
                    generated_noise = self.feedforward_model(
                        noisy_image,
                        timesteps.cast(paddle.float64),
                        text_embedding,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
            else:
                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()

                with paddle.no_grad():
                    generated_noise = self.feedforward_model(
                        noisy_image,
                        timesteps.cast(paddle.float64),
                        text_embedding,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample

                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).enable_gradient_checkpointing()

            # this assume that all teacher models use epsilon prediction (which is true for SDv1.5 and SDXL)
            generated_image = get_x0_from_noise(
                noisy_image.cast(paddle.float64),
                generated_noise.cast(paddle.float64),
                self.alphas_cumprod.cast(paddle.float64),
                timesteps,
            ).cast(paddle.float32)

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "text_embedding": text_embedding,
                    "pooled_text_embedding": pooled_text_embedding,
                    "uncond_embedding": uncond_embedding,
                    "real_train_dict": real_train_dict,
                    "unet_added_conditions": unet_added_conditions,
                    "uncond_unet_added_conditions": uncond_unet_added_conditions,
                }

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True, guidance_turn=False, generator_data_dict=generator_data_dict
                )
                self.guidance_model.requires_grad_(True)
                if isinstance(self.guidance_model, paddle.DataParallel):
                    self.guidance_model._layers.real_unet.requires_grad_(False)
                    self.guidance_model._layers.dummy_network.requires_grad_(False)
                else:
                    self.guidance_model.real_unet.requires_grad_(False)
                    self.guidance_model.dummy_network.requires_grad_(False)
            else:
                loss_dict = {}
                log_dict = {}

            if visual:
                decode_key = ["dmtrain_pred_real_image", "dmtrain_pred_fake_image"]

                with paddle.no_grad():
                    if compute_generator_gradient and not self.args.gan_alone:
                        for key in decode_key:
                            if self.use_fp16 and self.not_sdxl_vae:
                                log_dict[key + "_decoded"] = self.decode_image(
                                    log_dict[key].detach()[: self.num_visuals].half()
                                )
                            else:
                                log_dict[key + "_decoded"] = self.decode_image(
                                    log_dict[key].detach()[: self.num_visuals]
                                )

                    if self.use_fp16 and self.not_sdxl_vae:
                        log_dict["generated_image"] = self.decode_image(
                            generated_image[: self.num_visuals].detach().half()
                        )
                    else:
                        log_dict["generated_image"] = self.decode_image(generated_image[: self.num_visuals].detach())

                    if self.denoising:
                        if self.use_fp16 and self.not_sdxl_vae:
                            log_dict["original_clean_image"] = self.decode_image(
                                denoising_dict["images"].detach()[: self.num_visuals].half()
                            )
                        else:
                            log_dict["original_clean_image"] = self.decode_image(
                                denoising_dict["images"].detach()[: self.num_visuals]
                            )

            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "text_embedding": text_embedding.detach(),
                "pooled_text_embedding": pooled_text_embedding.detach(),
                "uncond_embedding": uncond_embedding.detach(),
                "real_train_dict": real_train_dict,
                "unet_added_conditions": unet_added_conditions,
                "uncond_unet_added_conditions": uncond_unet_added_conditions,
            }

            log_dict["denoising_timestep"] = timesteps

        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False, guidance_turn=True, guidance_data_dict=guidance_data_dict
            )
        return loss_dict, log_dict
