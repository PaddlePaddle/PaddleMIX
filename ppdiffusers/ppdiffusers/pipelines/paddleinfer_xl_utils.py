# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import re
from typing import List, Optional, Union

from ..image_processor import VaeImageProcessor
from ..loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ..models.paddleinfer_runtime import PaddleInferRuntimeModel  # noqa
from ..schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    PreconfigEulerAncestralDiscreteScheduler,
    PreconfigLMSDiscreteScheduler,
    UniPCMultistepScheduler,
)
from ..utils import is_paddle_available, logging, randn_tensor

__all__ = ["PaddleInferDiffusionXLPipelineMixin"]

if is_paddle_available():
    import paddle


logger = logging.get_logger(__name__)

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]
    # we must to tensor first!
    return paddle.to_tensor(tokens, dtype="int64"), paddle.to_tensor(weights, dtype="float32")


def get_unweighted_text_embeddings(
    pipe,
    text_input: paddle.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
    infer_op=None,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)

    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(
                input_ids=text_input_chunk,
            )[0]
            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = paddle.concat(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(
            input_ids=text_input,
        )[0]
    return text_embeddings


def get_weighted_text_embeddings(
    pipe,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    infer_op=None,
    **kwargs,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `1`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    # support bert tokenizer
    bos = pipe.tokenizer.bos_token_id if pipe.tokenizer.bos_token_id is not None else pipe.tokenizer.cls_token_id
    eos = pipe.tokenizer.eos_token_id if pipe.tokenizer.eos_token_id is not None else pipe.tokenizer.sep_token_id
    pad = pipe.tokenizer.pad_token_id

    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        no_boseos_middle=no_boseos_middle,
        infer_op=infer_op,
    )
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
            infer_op=infer_op,
        )
    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.mean(axis=[-2, -1])
        text_embeddings *= prompt_weights.unsqueeze(-1)
        text_embeddings *= (previous_mean / text_embeddings.mean(axis=[-2, -1])).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.mean(axis=[-2, -1])
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            uncond_embeddings *= (previous_mean / uncond_embeddings.mean(axis=[-2, -1])).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None


class PaddleInferDiffusionXLPipelineMixin:
    def prepare_infer_op_dict(self, infer_op_dict=None, **kwargs):
        # pass
        if infer_op_dict is None:
            infer_op_dict = {}
        new_infer_op_dict = {}
        return new_infer_op_dict

    def post_init(self, vae_scaling_factor=0.13025, vae_scale_factor=8, dtype="float32"):
        self.vae_scaling_factor = vae_scaling_factor
        self.vae_scale_factor = vae_scale_factor

        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        # self.dtype = dtype
        self.supported_scheduler = [
            "pndm",
            "lms",
            "preconfig-lms",
            "euler",
            "euler-ancestral",
            "preconfig-euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ]
        self.original_scheduler_config = self.scheduler.config

    @property
    def vae_encoder_num_channels(self):
        # if self.vae_encoder is None:
        #     return 3
        # return self.vae_encoder.model.get_input_info(0).shape[1]
        return 3

    @property
    def vae_decoder_num_latent_channels(self):
        # if self.vae_decoder is None:
        #     return 4
        # return self.vae_decoder.model.get_input_info(0).shape[1]
        return 4

    @property
    def unet_num_latent_channels(self):
        # return self.unet.model.get_input_info(0).shape[1]
        return 4

    @property
    def unet_hidden_states_dim(self):
        # return self.unet.model.get_input_info(2).shape[2]
        return 2048

    @property
    def text_encoder_hidden_states_dim(self):
        # if not hasattr(self, "text_encoder") or self.text_encoder is None:
        #     return 768
        # return self.text_encoder.model.get_output_info(0).shape[2]
        return 768

    def change_scheduler(self, scheduler_type="ddim"):
        scheduler_type = scheduler_type.lower()
        if scheduler_type == "pndm":
            scheduler = PNDMScheduler.from_config(self.original_scheduler_config, skip_prk_steps=True)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "preconfig-lms":
            scheduler = PreconfigLMSDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "preconfig-euler-ancestral":
            scheduler = PreconfigEulerAncestralDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "dpm-multi":
            scheduler = DPMSolverMultistepScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "dpm-single":
            scheduler = DPMSolverSinglestepScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "kdpm2-ancestral":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "kdpm2":
            scheduler = KDPM2DiscreteScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "unipc-multi":
            scheduler = UniPCMultistepScheduler.from_config(self.original_scheduler_config)
        elif scheduler_type == "ddim":
            scheduler = DDIMScheduler.from_config(
                self.original_scheduler_config,
                steps_offset=1,
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif scheduler_type == "ddpm":
            scheduler = DDPMScheduler.from_config(
                self.original_scheduler_config,
            )
        elif scheduler_type == "deis-multi":
            scheduler = DEISMultistepScheduler.from_config(
                self.original_scheduler_config,
            )
        else:
            raise ValueError(
                f"Scheduler of type {scheduler_type} doesn't exist! Please choose in {self.supported_scheduler}!"
            )
        self.scheduler = scheduler

    def get_timesteps(self, num_inference_steps, strength=1.0):
        if strength >= 1:
            return self.scheduler.timesteps.cast(self.dtype), num_inference_steps

        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :].cast(self.dtype)

        if hasattr(self.scheduler, "step_index_offset"):
            self.scheduler.step_index_offset = t_start * self.scheduler.order

        num_inference_steps = num_inference_steps - t_start
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )

        return timesteps, num_inference_steps

    def prepare_controlnet_cond(
        self,
        controlnet_cond,
        controlnet_conditioning_scale,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance=False,
    ):
        control_image = self.control_image_processor.preprocess(
            controlnet_cond,
            height=height,
            width=width,
        )
        if isinstance(controlnet_conditioning_scale, (float, int)):
            controlnet_conditioning_scale = paddle.to_tensor([controlnet_conditioning_scale] * 13, dtype=self.dtype)
        elif isinstance(controlnet_conditioning_scale, (list, tuple)):
            controlnet_conditioning_scale = paddle.to_tensor(controlnet_conditioning_scale, dtype=self.dtype)
        else:
            raise ValueError(
                f"`controlnet_conditioning_scale` has to be of type `float` or `int` or `list` or `tuple` but is {type(controlnet_conditioning_scale)}"
            )
        assert controlnet_conditioning_scale.shape[0] == 13
        image_batch_size = control_image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
        control_image = control_image.repeat_interleave(repeat_by, axis=0)
        if do_classifier_free_guidance:
            control_image = paddle.concat([control_image] * 2)
        return control_image, controlnet_conditioning_scale

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

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            timesteps (`List[int]`, *optional*):
                    Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                    timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                    must be `None`.

        Returns:
            `Tuple[paddle.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
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

    def encode_image(self, image, num_images_per_prompt, infer_op=None):
        dtype = self.dtype

        if not isinstance(image, paddle.Tensor):
            image = self.feature_extractor(image, return_tensors="pd").pixel_values

        image = image._to(dtype=dtype)
        image_embeds = self.image_encoder(
            image=image,
        )[0]
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, axis=0)
        uncond_image_embeds = paddle.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
        infer_op=None,
    ):
        shape = [
            batch_size,
            self.vae_decoder_num_latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        ]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.cast(dtype=self.dtype)
            image_latents = self._encode_vae_image(image, infer_op)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, dtype=self.dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents
            if str(noise.dtype).replace("paddle.", "") != self.dtype:
                noise = noise.cast(self.dtype)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        if len(outputs) == 1:
            outputs = latents
        return outputs

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        do_classifier_free_guidance,
        return_masked_image_latents=True,
        infer_op=None,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = paddle.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.cast(dtype=self.dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.tile([batch_size // mask.shape[0], 1, 1, 1])

        mask = paddle.concat([mask] * 2) if do_classifier_free_guidance else mask
        if not return_masked_image_latents:
            return mask

        masked_image = masked_image.cast(dtype=self.dtype)
        masked_image_latents = self._encode_vae_image(masked_image, infer_op)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.tile([batch_size // masked_image_latents.shape[0], 1, 1, 1])

        masked_image_latents = (
            paddle.concat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.cast(dtype=self.dtype)
        return mask, masked_image_latents

    def is_scheduler_support_step_index(self):
        kwargs_keys = set(inspect.signature(self.scheduler.step).parameters.keys())
        return "kwargs" in kwargs_keys or "step_index" in kwargs_keys

    def _encode_vae_image(self, image: paddle.Tensor, infer_op=None, **kwargs):
        image_latents = self.vae_encoder(
            sample=image,
        )[0]

        return self.vae_scaling_factor * image_latents

    def _decode_vae_latents(self, latents: paddle.Tensor, infer_op=None, **kwargs):
        images_vae = self.vae_decoder(
            latent_sample=latents,
        )[0]

        return images_vae

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = 256 * len(add_time_ids) + 1280
        expected_add_embed_dim = 2816
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = paddle.to_tensor(data=[add_time_ids], dtype=dtype)
        return add_time_ids

    def _get_add_time_ids_2(
        self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = 256 * len(add_time_ids) + 1280
        expected_add_embed_dim = 2816
        if expected_add_embed_dim > passed_add_embed_dim and expected_add_embed_dim - passed_add_embed_dim == 256:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif expected_add_embed_dim < passed_add_embed_dim and passed_add_embed_dim - expected_add_embed_dim == 256:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = paddle.to_tensor(data=[add_time_ids], dtype=dtype)
        add_neg_time_ids = paddle.to_tensor(data=[add_neg_time_ids], dtype=dtype)
        return add_time_ids, add_neg_time_ids

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if (
            callback_steps is None
            or callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}."
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f"`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, denoising_start=None):  # noqa
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - denoising_start * self.scheduler.config.num_train_timesteps
                )
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            return paddle.to_tensor(data=timesteps), len(timesteps)
        return timesteps, num_inference_steps - t_start

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        lora_scale: Optional[float] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`paddle.Tensoroptional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensoroptional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizer = self.tokenizer
        tokenizer_2 = self.tokenizer_2
        text_encoder = self.text_encoder
        text_encoder_2 = self.text_encoder_2
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            # for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            # 1. text encoder 1
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pd").input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pd").input_ids
            if (
                untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                and not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item()
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                )
            # prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids,
            )
            prompt_embeds_list.append(prompt_embeds[0])

            # 2. text encoder 2
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt_2, tokenizer_2)

            text_inputs = tokenizer_2(
                prompt_2,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer_2(prompt_2, padding="longest", return_tensors="pd").input_ids
            untruncated_ids = tokenizer_2(prompt_2, padding="longest", return_tensors="pd").input_ids
            if (
                untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                and not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item()
            ):
                removed_text = tokenizer_2.batch_decode(untruncated_ids[:, tokenizer_2.model_max_length - 1 : -1])
                logger.warning(
                    f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                )
            # prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
            prompt_embeds = self.text_encoder_2(
                input_ids=text_input_ids,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[1]
            # prompt_embeds = prompt_embeds.hidden_states[-2]
            # todo: check if this is correct
            prompt_embeds = prompt_embeds[0]
            prompt_embeds_list.append(prompt_embeds)

            # 3. Concatenate prompt embeddings
            prompt_embeds = paddle.concat(x=prompt_embeds_list, axis=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = paddle.zeros_like(x=prompt_embeds)
            negative_pooled_prompt_embeds = paddle.zeros_like(x=pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_embeds_list = []
            # 1. text_enocder 1
            if isinstance(self, TextualInversionLoaderMixin):
                negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pd"
            )
            # negative_prompt_embeds = text_encoder(uncond_input.input_ids, output_hidden_states=True)
            negative_prompt_embeds = text_encoder(
                input_ids=uncond_input.input_ids,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder

            # negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            # negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            # todo: check if this is correct
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds_list.append(negative_prompt_embeds)

            # 2. text_enocder 2
            if isinstance(self, TextualInversionLoaderMixin):
                negative_prompt = self.maybe_convert_prompt(negative_prompt_2, tokenizer_2)
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer_2(
                negative_prompt_2, padding="max_length", max_length=max_length, truncation=True, return_tensors="pd"
            )
            # negative_prompt_embeds = text_encoder(uncond_input.input_ids, output_hidden_states=True)
            negative_prompt_embeds = text_encoder_2(
                input_ids=uncond_input.input_ids,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder

            negative_pooled_prompt_embeds = negative_prompt_embeds[1]
            # negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            # todo: check if this is correct
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds_list.append(negative_prompt_embeds)

            # 3. combine the two embeddings
            negative_prompt_embeds = paddle.concat(x=negative_prompt_embeds_list, axis=-1)
        prompt_embeds = prompt_embeds.cast(dtype=self.dtype)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.cast(dtype=self.dtype)
            negative_prompt_embeds = negative_prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])
        pooled_prompt_embeds = pooled_prompt_embeds.tile(repeat_times=[1, num_images_per_prompt]).reshape(
            [bs_embed * num_images_per_prompt, -1]
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.tile(
                repeat_times=[1, num_images_per_prompt]
            ).reshape([bs_embed * num_images_per_prompt, -1])

        return (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)

    def _encode_prompt_old(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        infer_op=None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        **kwargs,
    ):
        if parse_prompt_type == "lpw":
            return self._encode_prompt_lpw(
                prompt,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_embeddings_multiples=max_embeddings_multiples,
                infer_op=infer_op,
                **kwargs,
            )
        elif parse_prompt_type == "raw":
            return self._encode_prompt_raw(
                prompt,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                infer_op=infer_op,
            )
        elif parse_prompt_type == "webui":
            raise NotImplementedError("`parse_prompt_type=webui` is not implemented yet.")

    def _encode_prompt_lpw(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Union[str, List[str]],
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        infer_op=None,
        max_embeddings_multiples: Optional[int] = 3,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None and negative_prompt_embeds is None:
            uncond_tokens: List[str] = None
            if do_classifier_free_guidance:
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=uncond_tokens,
                max_embeddings_multiples=max_embeddings_multiples,
                infer_op="raw",  # NOTE: we can't use zero copy!
                **kwargs,
            )

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def _encode_prompt_raw(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        infer_op=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids  # check

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids,
            )[0]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
            )
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input.input_ids,
            )[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if paddle.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="np")
            image, has_nsfw_concept = self.safety_checker(
                images=image.numpy(),
                clip_input=safety_checker_input.pixel_values.astype(self.dtype),
                infer_op="raw",
            )
            image = paddle.to_tensor(image, dtype=self.dtype)
        return image, has_nsfw_concept

    def prepare_extra_step_kwargs(self, generator, eta):  # noqa
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
