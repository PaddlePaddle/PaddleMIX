# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import inspect
import math
import os

import numpy as np
import paddle
import paddle.amp
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import AutoTokenizer, CLIPTextModel
from paddlenlp.utils.log import logger

from ppdiffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from ppdiffusers.training_utils import freeze_params

from .lcm_scheduler import LCMScheduler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas[timesteps] * sample - sigmas[timesteps] * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.take_along_axis(t, axis=-1)
    return out.reshape([b, *((1,) * (len(x_shape) - 1))])


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = paddle.to_tensor(self.ddim_timesteps).cast("int64")
        self.ddim_alpha_cumprods = paddle.to_tensor(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = paddle.to_tensor(self.ddim_alpha_cumprods_prev)

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


def get_guidance_scale_embedding(w, embedding_dim=512, dtype=paddle.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`paddle.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `paddle.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = paddle.log(paddle.to_tensor(10000.0)) / (half_dim - 1)
    emb = paddle.exp(paddle.arange(half_dim, dtype=dtype) * -emb)
    emb = w.cast(dtype=dtype)[:, None] * emb[None, :]
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = paddle.concat(emb, paddle.zeros([emb.shape[0], 1]), axis=-1)
    assert emb.shape == [w.shape[0], embedding_dim]
    return emb


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    timesteps=None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(axis=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(axis=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class LCMModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.is_lora = model_args.is_lora
        tokenizer_name_or_path = (
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
        )
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.vae_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "vae")
        )
        text_encoder_name_or_path = (
            model_args.text_encoder_name_or_path
            if model_args.text_encoder_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder")
        )
        teacher_unet_name_or_path = (
            model_args.teacher_unet_name_or_path
            if model_args.teacher_unet_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "unet")
        )
        # init model and tokenizer
        tokenizer_kwargs = {}
        if model_args.model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = model_args.model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)
        self.teacher_unet = UNet2DConditionModel.from_pretrained(teacher_unet_name_or_path)
        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.parameters())
        freeze_params(self.teacher_unet.parameters())
        self.vae.eval()
        self.text_encoder.eval()
        self.teacher_unet.eval()

        if self.is_lora:
            self.unet = UNet2DConditionModel.from_pretrained(teacher_unet_name_or_path)
            self.unet.train()
            from paddlenlp.peft import LoRAConfig, LoRAModel

            lora_config = LoRAConfig(
                r=model_args.lora_rank,
                target_modules=[
                    ".*to_q.*",
                    ".*to_k.*",
                    ".*to_v.*",
                    ".*to_out.0.*",
                    ".*proj_in.*",
                    ".*proj_out.*",
                    ".*ff.net.0.proj.*",
                    ".*ff.net.2.*",
                    ".*conv1.*",
                    ".*conv2.*",
                    ".*conv_shortcut.*",
                    ".*downsamplers.0.conv.*",
                    ".*upsamplers.0.conv.*",
                    ".*time_emb_proj.*",
                ],
                merge_weights=False,  # make sure we donot merge weights
            )
            self.unet.config.tensor_parallel_degree = 1
            self.unet = LoRAModel(self.unet, lora_config)
            self.reset_lora_parameters(self.unet, init_lora_weights=True)
            self.unet.mark_only_lora_as_trainable()
            self.unet.print_trainable_parameters()
        else:
            # 8. Create online (`unet`) student U-Nets. This will be updated by the optimizer (e.g. via backpropagation.)
            # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
            if self.teacher_unet.config.time_cond_proj_dim is None:
                self.teacher_unet.config["time_cond_proj_dim"] = self.model_args.unet_time_cond_proj_dim
            self.unet = UNet2DConditionModel(**self.teacher_unet.config)
            # load teacher_unet weights into unet
            self.unet.load_dict(self.teacher_unet.state_dict())
            self.unet.train()

            # 9. Create target (`ema_unet`) student U-Net parameters. This will be updated via EMA updates (polyak averaging).
            # Initialize from unet
            self.target_unet = UNet2DConditionModel(**self.teacher_unet.config)
            self.target_unet.load_dict(self.unet.state_dict())
            self.target_unet.train()
            freeze_params(self.target_unet.parameters())

        # init noise_scheduler and eval_scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(vae_name_or_path.replace("vae", "scheduler"))
        self.alpha_schedule = paddle.sqrt(self.noise_scheduler.alphas_cumprod)
        self.sigma_schedule = paddle.sqrt(1 - self.noise_scheduler.alphas_cumprod)
        self.solver = DDIMSolver(
            self.noise_scheduler.alphas_cumprod.numpy(),
            timesteps=self.noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=model_args.num_ddim_timesteps,
        )
        self.eval_scheduler = LCMScheduler.from_pretrained(vae_name_or_path.replace("vae", "scheduler"))

        uncond_input_ids = self.tokenizer(
            [""], return_tensors="pd", padding="max_length", max_length=self.tokenizer.model_max_length
        ).input_ids
        self.uncond_prompt_embeds = self.text_encoder(uncond_input_ids)[0]
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        self.vae.eval()
        self.text_encoder.eval()
        self.teacher_unet.eval()

        with paddle.no_grad():
            latents = []
            for i in range(0, pixel_values.shape[0], self.model_args.vae_encode_max_batch_size):
                latents.append(
                    self.vae.encode(
                        pixel_values[i : i + self.model_args.vae_encode_max_batch_size]
                    ).latent_dist.sample()
                )
            latents = paddle.concat(latents, axis=0)
            latents = latents * self.vae.config.scaling_factor

            # 20.4.8. Prepare prompt embeds and unet_added_conditions
            prompt_embeds = self.text_encoder(input_ids)[0]

        # Sample noise that we'll add to the latents
        noise = paddle.randn(latents.shape, dtype=latents.dtype)
        bsz = latents.shape[0]

        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = self.noise_scheduler.config.num_train_timesteps // self.model_args.num_ddim_timesteps
        index = paddle.randint(0, self.model_args.num_ddim_timesteps, (bsz,), dtype="int64")
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = paddle.where(timesteps < 0, paddle.zeros_like(timesteps), timesteps)

        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)

        # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
        w = (self.model_args.w_max - self.model_args.w_min) * paddle.rand((bsz,)) + self.model_args.w_min

        if not self.model_args.is_lora:
            w_embedding = get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim)
        else:
            w_embedding = None
        w = w.reshape([bsz, 1, 1, 1])

        # 20.4.9. Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
        noise_pred = self.unet(
            noisy_model_input,
            start_timesteps,
            timestep_cond=w_embedding,
            encoder_hidden_states=prompt_embeds,
        ).sample

        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            self.noise_scheduler.config.prediction_type,
            self.alpha_schedule,
            self.sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        # 20.4.10. Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
        # noisy_latents with both the conditioning embedding c and unconditional embedding 0
        # Get teacher model prediction on noisy_latents and conditional embedding
        with paddle.no_grad():
            with paddle.amp.auto_cast():
                cond_teacher_output = self.teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample
                cond_pred_x0 = predicted_origin(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule,
                )

                # Get teacher model prediction on noisy_latents and unconditional embedding
                uncond_teacher_output = self.teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=self.uncond_prompt_embeds.expand([bsz, -1, -1]),
                ).sample
                uncond_pred_x0 = predicted_origin(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule,
                )

                # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

        # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
        with paddle.no_grad():
            with paddle.amp.auto_cast():
                module = self.unet if self.is_lora else self.target_unet
                target_noise_pred = module(
                    x_prev,
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                ).sample
            pred_x_0 = predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
                self.noise_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        # 20.4.13. Calculate loss
        if self.model_args.loss_type == "l2":
            loss = F.mse_loss(
                model_pred.cast(dtype=paddle.float32), target.cast(dtype=paddle.float32), reduction="mean"
            )
        elif self.model_args.loss_type == "huber":
            loss = paddle.mean(
                paddle.sqrt(
                    (model_pred.cast(dtype=paddle.float32) - target.cast(dtype=paddle.float32)) ** 2
                    + self.model_args.huber_c**2
                )
                - self.model_args.huber_c
            )
        return loss

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, max_batch=8, **kwargs):
        self.eval()
        if pixel_values.shape[0] > max_batch:
            pixel_values = pixel_values[:max_batch]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        height=256,
        width=256,
        max_batch=8,
        timesteps=None,
        **kwargs,
    ):
        if self.is_lora:
            guidance_scale = 1.0
        else:
            guidance_scale = 7.5
        guidance_rescale = 0.0
        num_inference_steps = self.model_args.num_inference_steps

        #########################################################
        self.eval()
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )
        # only log max_batch image
        if input_ids.shape[0] > max_batch:
            input_ids = input_ids[:max_batch]

        prompt_embeds = self.text_encoder(input_ids)[0]

        do_classifier_free_guidance = guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
        if do_classifier_free_guidance:
            prompt_embeds = paddle.concat(
                [self.uncond_prompt_embeds.expand([prompt_embeds.shape[0], -1, -1]), prompt_embeds]
            )

        batch_size = prompt_embeds.shape[0]
        timesteps, num_inference_steps = retrieve_timesteps(self.eval_scheduler, num_inference_steps, timesteps)
        shape = [batch_size, 4, height // self.vae_scale_factor, width // self.vae_scale_factor]
        latents = paddle.randn(shape, dtype=prompt_embeds.dtype) * self.eval_scheduler.init_noise_sigma

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = paddle.to_tensor([guidance_scale - 1]).tile(
                [
                    batch_size,
                ]
            )
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.eval_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # vae decode
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.unet.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self):
        self.set_recompute(True)

    def set_xformers(self, use_xformers=False):
        if not use_xformers:
            if hasattr(self.unet, "set_default_attn_processor"):
                self.unet.set_default_attn_processor()
            if hasattr(self.vae, "set_default_attn_processor"):
                self.vae.set_default_attn_processor()

    @paddle.no_grad()
    def reset_lora_parameters(self, unet, init_lora_weights=True):
        if init_lora_weights is False:
            return
        for name, module in unet.named_sublayers(include_self=True):
            module_name = module.__class__.__name__.lower()
            if module_name in ["loralinear"]:
                if init_lora_weights is True:
                    # initialize A the same way as the default for nn.Linear and B to zero
                    # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                    tmp_tensor = paddle.zeros(module.lora_A.T.shape, dtype=module.lora_A.dtype)
                    nn.init.kaiming_uniform_(tmp_tensor, a=math.sqrt(5))
                    module.lora_A.set_value(tmp_tensor.T)
                    del tmp_tensor
                    logger.info(f"Initialized {name}'s LoRA parameters with Kaiming uniform init!")
                elif init_lora_weights.lower() == "gaussian":
                    tmp_tensor = paddle.zeros(module.lora_A.T.shape, dtype=module.lora_A.dtype)
                    nn.init.normal_(tmp_tensor, std=1 / self.r)
                    module.lora_A.set_value(tmp_tensor.T)
                    del tmp_tensor
                    logger.info(f"Initialized {name}'s LoRA parameters with Gaussian init!")
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights}!")
                nn.init.zeros_(module.lora_B)
            elif module_name in ["loraconv2d"]:
                if init_lora_weights is True:
                    # initialize A the same way as the default for nn.Linear and B to zero
                    # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                    nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                    logger.info(f"Initialized {name}'s LoRA parameters with Kaiming uniform init!")
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(module.lora_A, std=1 / self.r)
                    logger.info(f"Initialized {name}'s LoRA parameters with Gaussian init!")
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights}!")
                nn.init.zeros_(module.lora_B)
