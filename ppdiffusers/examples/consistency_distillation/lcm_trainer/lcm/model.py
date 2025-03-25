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
import contextlib
import copy
import inspect
import math
import os

import numpy as np
import paddle
import paddle.amp
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.utils.log import logger

from ppdiffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from ppdiffusers.training_utils import freeze_params
from ppdiffusers.transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from ppdiffusers.utils.initializer_utils import reset_initialized_parameter

from .lcm_scheduler import LCMScheduler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )
    return pred_epsilon


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


@paddle.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.copy_(rate * targ + (1 - rate) * src, False)


class LCMModel(nn.Layer):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args
        self.is_lora = model_args.is_lora
        self.is_sdxl = model_args.is_sdxl

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

        # text encoder 1
        tokenizer_name_or_path = (
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
        )
        # init model and tokenizer
        tokenizer_kwargs = {}
        if model_args.model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = model_args.model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path, dtype="float32")
        freeze_params(self.text_encoder.parameters())

        # text encoder 2
        if self.is_sdxl:
            tokenizer_2_name_or_path = (
                model_args.tokenizer_2_name
                if model_args.tokenizer_2_name is not None
                else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer_2")
            )
            text_encoder_2_name_or_path = (
                model_args.text_encoder_2_name_or_path
                if model_args.text_encoder_2_name_or_path is not None
                else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder_2")
            )
            # init model and tokenizer
            self.tokenizer_2 = AutoTokenizer.from_pretrained(tokenizer_2_name_or_path, **tokenizer_kwargs)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                text_encoder_2_name_or_path, dtype="float32"
            )
            freeze_params(self.text_encoder_2.parameters())

        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path, paddle_dtype=paddle.float32)
        self.teacher_unet = UNet2DConditionModel.from_pretrained(
            teacher_unet_name_or_path, paddle_dtype=paddle.float32
        )
        freeze_params(self.vae.parameters())
        freeze_params(self.teacher_unet.parameters())
        self.vae.eval()
        self.text_encoder.eval()
        self.teacher_unet.eval()

        if self.is_lora:
            self.unet = UNet2DConditionModel.from_pretrained(teacher_unet_name_or_path, paddle_dtype=paddle.float32)
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
                merge_weights=False,  # make sure we do not merge weights
            )
            self.unet.config.tensor_parallel_degree = 1
            self.unet = LoRAModel(self.unet, lora_config)
            self.reset_lora_parameters(self.unet, init_lora_weights=False)
            self.unet.mark_only_lora_as_trainable()
            self.unet.print_trainable_parameters()
            self.time_cond_proj_dim = None
        else:
            # Create online (`unet`) student U-Nets. This will be updated by the optimizer (e.g. via backpropagation.)
            # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
            time_cond_proj_dim = (
                self.teacher_unet.config.time_cond_proj_dim
                if self.teacher_unet.config.time_cond_proj_dim is not None
                else self.model_args.unet_time_cond_proj_dim
            )
            self.unet = UNet2DConditionModel.from_config(
                self.teacher_unet.config,
                time_cond_proj_dim=time_cond_proj_dim,
            )
            # initialize the `time_embedding.cond_proj` like `torch.nn.Linear``
            reset_initialized_parameter(self.unet.time_embedding.cond_proj)
            # load teacher_unet weights into unet
            self.unet.load_dict(self.teacher_unet.state_dict())
            self.unet.train()

            # Create target (`ema_unet`) student U-Net parameters. This will be updated via EMA updates (polyak averaging).
            # Initialize from unet
            self.target_unet = UNet2DConditionModel.from_config(
                self.teacher_unet.config,
                time_cond_proj_dim=time_cond_proj_dim,
            )
            self.target_unet.load_dict(self.unet.state_dict())
            self.target_unet.train()
            freeze_params(self.target_unet.parameters())
            self.time_cond_proj_dim = time_cond_proj_dim

        # init noise_scheduler and eval_scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(vae_name_or_path.replace("vae", "scheduler"))
        self.alpha_schedule = paddle.sqrt(self.noise_scheduler.alphas_cumprod)
        self.sigma_schedule = paddle.sqrt(1 - self.noise_scheduler.alphas_cumprod)
        self.solver = DDIMSolver(
            self.noise_scheduler.alphas_cumprod.numpy(),
            timesteps=self.noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=model_args.num_ddim_timesteps,
        )

        # set this attr for eval
        self.eval_scheduler = LCMScheduler.from_pretrained(vae_name_or_path.replace("vae", "scheduler"))
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if not self.is_sdxl:
            # Create uncond embeds for classifier free guidance
            uncond_input_ids = self.tokenizer(
                [""],
                return_tensors="pd",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            self.uncond_prompt_embeds = self.text_encoder(uncond_input_ids)[0]

        if training_args.bf16 or training_args.fp16:
            self.autocast_smart_context_manager = contextlib.nullcontext
        else:
            self.autocast_smart_context_manager = paddle.amp.auto_cast

    @paddle.no_grad()
    def compute_embeddings(self, input_ids=None, input_ids_2=None, add_time_ids=None, height=1024, width=1024):
        if self.is_sdxl:
            assert input_ids_2 is not None
            if add_time_ids is None:
                add_time_ids = self._get_add_time_ids((height, width), (0, 0), (height, width), dtype="float32").tile(
                    [input_ids.shape[0], 1]
                )
            prompt_embeds_list = []
            for text_inputs_ids, text_encoder in zip(
                [input_ids, input_ids_2], [self.text_encoder, self.text_encoder_2]
            ):
                text_encoder.eval()
                prompt_embeds = text_encoder(
                    text_inputs_ids,
                    output_hidden_states=True,
                )
                prompt_embeds_list.append(prompt_embeds.hidden_states[-2])
            # text encoder 2 pooled output
            add_text_embeds = prompt_embeds[0]
            prompt_embeds = paddle.concat(prompt_embeds_list, axis=-1)
            unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        else:
            prompt_embeds = self.text_encoder(input_ids)[0]
            unet_added_cond_kwargs = {}
        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    def forward(self, input_ids=None, pixel_values=None, input_ids_2=None, add_time_ids=None, **kwargs):
        self.vae.eval()
        self.teacher_unet.eval()

        encoded_text = self.compute_embeddings(
            input_ids=input_ids,
            input_ids_2=input_ids_2,
            add_time_ids=add_time_ids,
        )

        with paddle.no_grad():
            # encode pixel values with batch size of at most vae_encode_batch_size
            latents = []
            for i in range(0, pixel_values.shape[0], self.model_args.vae_encode_batch_size):
                latents.append(
                    self.vae.encode(pixel_values[i : i + self.model_args.vae_encode_batch_size]).latent_dist.sample()
                )
            latents = paddle.concat(latents, axis=0)
            latents = latents * self.vae.config.scaling_factor
            bsz = latents.shape[0]

        # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
        # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
        topk = self.noise_scheduler.config.num_train_timesteps // self.model_args.num_ddim_timesteps
        index = paddle.randint(0, self.model_args.num_ddim_timesteps, (bsz,), dtype="int64")
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = paddle.where(timesteps < 0, paddle.zeros_like(timesteps), timesteps)

        # 3. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps, timestep_scaling=self.model_args.timestep_scaling_factor
        )
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(
            timesteps, timestep_scaling=self.model_args.timestep_scaling_factor
        )
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
        # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noise = paddle.randn(latents.shape, dtype=latents.dtype)
        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)

        # 5. Sample a random guidance scale w from U[w_min, w_max]
        # Note that for LCM-LoRA distillation it is not necessary to use a guidance scale embedding
        w = (self.model_args.w_max - self.model_args.w_min) * paddle.rand((bsz,)) + self.model_args.w_min
        if self.time_cond_proj_dim is None:
            w_embedding = None
        else:
            w_embedding = get_guidance_scale_embedding(w, embedding_dim=self.time_cond_proj_dim)
        w = w.reshape([bsz, 1, 1, 1])

        # 6. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text.pop("prompt_embeds")

        # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
        noise_pred = self.unet(
            noisy_model_input,
            start_timesteps,
            timestep_cond=w_embedding,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=encoded_text,
        ).sample

        pred_x_0 = get_predicted_original_sample(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            self.noise_scheduler.config.prediction_type,
            self.alpha_schedule,
            self.sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
        # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
        # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
        # solver timestep.
        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                cond_teacher_output = self.teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=encoded_text,
                ).sample
                cond_pred_x0 = get_predicted_original_sample(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule,
                )
                cond_pred_noise = get_predicted_noise(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule,
                )

                # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                if self.is_sdxl:
                    uncond_prompt_embeds = paddle.zeros_like(prompt_embeds)
                    uncond_added_conditions = copy.deepcopy(encoded_text)
                    uncond_added_conditions["text_embeds"] = paddle.zeros_like(encoded_text["text_embeds"])
                else:
                    uncond_prompt_embeds = self.uncond_prompt_embeds.expand([bsz, -1, -1])
                    uncond_added_conditions = {}
                uncond_teacher_output = self.teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=uncond_prompt_embeds,
                    added_cond_kwargs=uncond_added_conditions,
                ).sample
                uncond_pred_x0 = get_predicted_original_sample(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule,
                )
                uncond_pred_noise = get_predicted_noise(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule,
                )

                # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                # augmented PF-ODE trajectory (solving backward in time)
                # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

        # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
        # Note that we do not use a separate target network for LCM-LoRA distillation.
        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                module = self.unet if self.is_lora else self.target_unet
                target_noise_pred = module(
                    x_prev,
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=encoded_text,
                ).sample
            pred_x_0 = get_predicted_original_sample(
                target_noise_pred,
                timesteps,
                x_prev,
                self.noise_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        # 10. Calculate loss
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

    def on_train_batch_end(self):
        if not self.is_lora:
            update_ema(self.target_unet.parameters(), self.unet.parameters(), self.model_args.ema_decay)

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.weight.shape[0]
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = paddle.to_tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        height=512,
        width=512,
        max_batch=4,
        timesteps=None,
        seed=42,
        unet=None,
        input_ids_2=None,
        prompt=None,
        **kwargs,
    ):
        orig_rng_state = None
        if seed is not None and seed > 0:
            orig_rng_state = paddle.get_rng_state()
            paddle.seed(seed)

        if self.time_cond_proj_dim is None:
            guidance_scale = 1.0
        else:
            guidance_scale = 5.0
        guidance_rescale = 0.0
        num_inference_steps = self.model_args.num_inference_steps

        if prompt is not None:
            input_ids = self.tokenizer(
                [prompt] * max_batch,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            ).input_ids
            if self.is_sdxl:
                input_ids_2 = self.tokenizer_2(
                    [prompt] * max_batch,
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pd",
                ).input_ids
        #########################################################
        self.eval()
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )
        # only log max_batch image
        if input_ids.shape[0] > max_batch:
            input_ids = input_ids[:max_batch]
        if input_ids_2 is not None and input_ids_2.shape[0] > max_batch:
            input_ids_2 = input_ids_2[:max_batch]

        # choose unet to infer
        unet = unet or self.unet

        encoded_text = self.compute_embeddings(
            input_ids=input_ids,
            input_ids_2=input_ids_2,
            height=height,
            width=width,
        )
        prompt_embeds = encoded_text.pop("prompt_embeds")

        shape = [prompt_embeds.shape[0], 4, height // self.vae_scale_factor, width // self.vae_scale_factor]
        latents = paddle.randn(shape, dtype=prompt_embeds.dtype) * self.eval_scheduler.init_noise_sigma

        do_classifier_free_guidance = guidance_scale > 1 and self.time_cond_proj_dim is None
        if do_classifier_free_guidance:
            prompt_embeds = paddle.concat([paddle.zeros_like(prompt_embeds), prompt_embeds], axis=0)
            if self.is_sdxl:
                text_embeds = encoded_text.pop("text_embeds")
                encoded_text["text_embeds"] = paddle.concat([paddle.zeros_like(text_embeds), text_embeds], axis=0)
                time_ids = encoded_text.pop("time_ids")
                encoded_text["time_ids"] = paddle.concat([time_ids, time_ids], axis=0)

        timesteps, num_inference_steps = retrieve_timesteps(self.eval_scheduler, num_inference_steps, timesteps)

        timestep_cond = None
        if self.time_cond_proj_dim is not None:
            guidance_scale_tensor = paddle.to_tensor([guidance_scale - 1]).tile(
                [
                    prompt_embeds.shape[0],
                ]
            )
            timestep_cond = get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.time_cond_proj_dim)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                added_cond_kwargs=encoded_text,
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
        if orig_rng_state is not None:
            paddle.set_rng_state(orig_rng_state)
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
            if module_name in ["loralinear", "loraconv2d"]:
                if init_lora_weights is True:
                    # initialize A the same way as the default for nn.Linear and B to zero
                    # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                    nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5), reverse=module_name == "loralinear")
                    logger.info(f"Initialized {name}'s LoRA parameters with Kaiming uniform init!")
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(module.lora_A, std=1 / self.r)
                    logger.info(f"Initialized {name}'s LoRA parameters with Gaussian init!")
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights}!")
                nn.init.zeros_(module.lora_B)
