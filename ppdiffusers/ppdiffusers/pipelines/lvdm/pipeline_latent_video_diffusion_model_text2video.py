# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle
from einops import rearrange
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...models import LVDMAutoencoderKL, LVDMUNet3DModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import deprecate, logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from . import VideoPipelineOutput
from .video_save import save_results

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
         >>> import paddle
         >>> from ppdiffusers import LVDMTextToVideoPipeline
         >>> pipe = LVDMTextToVideoPipeline.from_pretrained("westfish/lvdm_text2video_orig_webvid_2m")
         >>> seed = 2013
         >>> generator = paddle.Generator().manual_seed(seed)
         >>> samples = pipe(
                    prompt="cutting in kitchen",
                    num_frames=16,
                    height=256,
                    width=256,
                    num_inference_steps=50,
                    generator=generator,
                    guidance_scale=15
                    eta=1,
                    save_dir='.',
                    save_name='ddim_lvdm_text_to_video_ucf',
                    encoder_type='2d',
                    scale_factor=0.18215,
                    shift_factor=0,
            )
        >>> prompt = "cliff diving"
        >>> image = pipe(prompt).video[0]
        ```
"""


def split_video_to_clips(video, clip_length, drop_left=True):
    video_length = video.shape[2]
    shape = video.shape
    if video_length % clip_length != 0 and drop_left:
        video = video[:, :, : video_length // clip_length * clip_length, :, :]
        print(f"[split_video_to_clips] Drop frames from {shape} to {video.shape}")
    nclips = video_length // clip_length
    clips = rearrange(video, "b c (nc cl) h w -> (b nc) c cl h w", cl=clip_length, nc=nclips)
    return clips


def merge_clips_to_videos(clips, bs):
    nclips = clips.shape[0] // bs
    video = rearrange(clips, "(b nc) c t h w -> b c (nc t) h w", nc=nclips)
    return video


class LVDMTextToVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Latent Video Diffusion Model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`LVDMAutoencoderKL`]):
            Autoencoder Model to encode and decode videos to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`LVDMUNet3DModel`]): 3D conditional U-Net architecture to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`]
            or [`DPMSolverMultistepScheduler`].
    """

    def __init__(
        self,
        vae: LVDMAutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: LVDMUNet3DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        # self.encoder_type = '2d'
        # self.scale_factor = 0.18215
        # self.shift_factor = 0

    @paddle.no_grad()
    def decode(self, z, **kwargs):
        z = 1.0 / kwargs["scale_factor"] * z - kwargs["shift_factor"]
        results = self.vae.decode(z).sample
        return results

    @paddle.no_grad()
    def overlapped_decode(self, z, max_z_t=None, overlap_t=2, predict_cids=False, force_not_quantize=False):
        if max_z_t is None:
            max_z_t = z.shape[2]
        assert max_z_t > overlap_t
        max_x_t = max_z_t * 4
        drop_r = overlap_t // 2
        drop_l = overlap_t - drop_r
        drop_r_x = drop_r * 4
        drop_l_x = drop_l * 4
        start = 0
        end = max_z_t
        zs = []
        while start <= z.shape[2]:
            zs.append(z[:, :, start:end, :, :])
            start += max_z_t - overlap_t
            end += max_z_t - overlap_t
        reses = []
        for i, z_ in enumerate(zs):
            if i == 0:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, : max_x_t - drop_r_x, :, :]
            elif i == len(zs) - 1:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, drop_l_x:, :, :]
            else:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[
                    :, :, drop_l_x : max_x_t - drop_r_x, :, :
                ]
            reses.append(res)
        results = paddle.concat(x=reses, axis=2)
        return results

    @paddle.no_grad()
    def decode_first_stage_2DAE_video(self, z, decode_bs=16, return_cpu=True, **kwargs):
        b, _, t, _, _ = z.shape
        z = rearrange(z, "b c t h w -> (b t) c h w")
        if decode_bs is None:
            results = self.decode(z, **kwargs)
        else:
            z = paddle.split(x=z, num_or_sections=z.shape[0] // decode_bs, axis=0)
            if return_cpu:
                results = paddle.concat(x=[self.decode(z_, **kwargs).cpu() for z_ in z], axis=0)
            else:
                results = paddle.concat(x=[self.decode(z_, **kwargs) for z_ in z], axis=0)
        results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()
        return results

    @paddle.no_grad()
    def decode_latents(
        self,
        z,
        decode_bs=16,
        return_cpu=True,
        bs=None,
        decode_single_video_allframes=False,
        max_z_t=None,
        overlapped_length=0,
        **kwargs
    ):
        b, _, t, _, _ = z.shape
        if kwargs["encoder_type"] == "2d" and z.dim() == 5:
            return self.decode_first_stage_2DAE_video(z, decode_bs=decode_bs, return_cpu=return_cpu, **kwargs)
        if decode_single_video_allframes:
            z = paddle.split(x=z, num_or_sections=z.shape[0] // 1, axis=0)
            cat_dim = 0
        elif max_z_t is not None:
            if kwargs["encoder_type"] == "3d":
                z = paddle.split(x=z, num_or_sections=z.shape[2] // max_z_t, axis=2)
                cat_dim = 2
            if kwargs["encoder_type"] == "2d":
                z = paddle.split(x=z, num_or_sections=z.shape[0] // max_z_t, axis=0)
                cat_dim = 0
        # elif self.split_clips and self.downfactor_t is not None or self.clip_length is not None and self.downfactor_t is not None and z.shape[
        #     2
        #     ] > self.clip_length // self.downfactor_t and self.encoder_type == '3d':
        #     split_z_t = self.clip_length // self.downfactor_t
        #     print(f'split z ({z.shape}) to length={split_z_t} clips')
        #     z = split_video_to_clips(z, clip_length=split_z_t, drop_left=True)
        #     if bs is not None and z.shape[0] > bs:
        #         print(f'split z ({z.shape}) to bs={bs}')
        #         z = paddle.split(x=z, num_or_sections=z.shape[0] // bs, axis=0)
        #         cat_dim = 0
        paddle.device.cuda.empty_cache()
        if isinstance(z, tuple):
            zs = [self.decode(z_, **kwargs).cpu() for z_ in z]
            results = paddle.concat(x=zs, axis=cat_dim)
        elif isinstance(z, paddle.Tensor):
            results = self.decode(z, **kwargs)
        else:
            raise ValueError
        # if self.split_clips and self.downfactor_t is not None:
        #     results = merge_clips_to_videos(results, bs=b)
        return results

    @paddle.no_grad()
    def paddle_to_np(self, x):
        sample = x.detach().cpu()
        if sample.dim() == 5:
            sample = sample.transpose(perm=[0, 2, 3, 4, 1])
        else:
            sample = sample.transpose(perm=[0, 2, 3, 1])

        if isinstance("uint8", paddle.dtype):
            dtype = "uint8"
        elif isinstance("uint8", str) and "uint8" not in ["cpu", "cuda", "ipu", "xpu"]:
            dtype = "uint8"
        elif isinstance("uint8", paddle.Tensor):
            dtype = "uint8".dtype
        else:
            dtype = ((sample + 1) * 127.5).clip(min=0, max=255).dtype
        sample = ((sample + 1) * 127.5).clip(min=0, max=255).cast(dtype)

        sample = sample.numpy()
        return sample

    def _encode_prompt(
        self,
        prompt,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
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
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

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

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.cast(self.text_encoder.dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_videos_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_videos_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
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

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.cast(self.text_encoder.dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_videos_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_videos_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
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

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, generator, latents=None
    ):
        shape = [batch_size, num_channels_latents, num_frames, height // 8, width // 8]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        save_dir=None,
        save_name=None,
        num_frames: Optional[int] = 16,
        encoder_type="2d",
        scale_factor=0.18215,
        shift_factor=0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated video frame.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated video frame.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a VideoPipelineOutput instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in ppdiffusers.cross_attention.
            save_dir (`str` or `List[str]`, *optional*):
                If provided, will save videos generated to *save_dir*. Otherwise will save them to the current path.
            save_name (`str` or `List[str]`, *optional*):
                If provided, will save videos generated to *save_name*.
            num_frames (`int`, *optional*, defaults to 16):
                Number of frames of the video. If None, will generate 16 frames per video.
            encoder_type (`str`, *optional*, defaults to `"2d"`):
                If provided, will use the specified encoder to generate the video, chosen from [`2d`, `3d`].
            scale_factor (`float`, *optional*, defaults to 0.18215):
                Scale factor for the generated video.
            shift_factor (`float`, *optional*, defaults to 0):
                Shift factor for the generated video.
        Examples:
        Returns:
            [`VideoPipelineOutput`] or `tuple`: [`VideoPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timesteps=t,
                    context=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        all_videos = []
        extra_decode_kwargs = {
            "encoder_type": encoder_type,
            "scale_factor": scale_factor,
            "shift_factor": shift_factor,
        }
        sampled_videos = self.decode_latents(latents, decode_bs=1, return_cpu=False, **extra_decode_kwargs)
        all_videos.append(self.paddle_to_np(sampled_videos))
        all_videos = np.concatenate(all_videos, axis=0)

        # return sampled_videos
        videos_frames = []
        for idx in range(sampled_videos.shape[0]):
            video = sampled_videos[idx]
            video_frames = []
            for fidx in range(video.shape[1]):
                frame = video[:, fidx]
                frame = (frame / 2 + 0.5).clip(0, 1)
                frame = frame.transpose([1, 2, 0]).astype("float32").numpy()
                if output_type == "pil":
                    frame = self.numpy_to_pil(frame)
                video_frames.append(frame)
            videos_frames.append(video_frames)

        if not save_name:
            save_name = "defaul_video"
        if not save_dir:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)
        save_results(all_videos, save_dir=save_dir, save_name=save_name, save_fps=8)
        return VideoPipelineOutput(frames=videos_frames, samples=sampled_videos)
