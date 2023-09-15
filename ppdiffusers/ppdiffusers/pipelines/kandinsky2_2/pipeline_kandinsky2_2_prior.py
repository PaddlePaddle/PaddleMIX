# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import paddle
import paddlenlp
import PIL

from ...models import PriorTransformer
from ...schedulers import UnCLIPScheduler
from ...utils import logging, randn_tensor, replace_example_docstring
from ..kandinsky import KandinskyPriorPipelineOutput
from ..pipeline_utils import DiffusionPipeline

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from ppdiffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
        >>> import paddle

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
        >>> prompt = "red cat, 4k photo"
        >>> image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()

        >>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... ).images
        >>> image[0].save("cat.png")
        ```
"""
EXAMPLE_INTERPOLATE_DOC_STRING = """
    Examples:
        ```py
        >>> from ppdiffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
        >>> from ppdiffusers.utils import load_image
        >>> import PIL
        >>> import paddle
        >>> from from paddle.vision import transforms

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", paddle_dtype=paddle.float16
        ... )
        >>> img1 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )
        >>> img2 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/starry_night.jpeg"
        ... )
        >>> images_texts = ["a cat", img1, img2]
        >>> weights = [0.3, 0.3, 0.4]
        >>> out = pipe_prior.interpolate(images_texts, weights)
        >>> pipe = KandinskyV22Pipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder", paddle_dtype=paddle.float16
        ... )
        >>> image = pipe(
        ...     image_embeds=out.image_embeds,
        ...     negative_image_embeds=out.negative_image_embeds,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("starry_cat.png")
        ```
"""


class KandinskyV22PriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        image_processor ([`CLIPImageProcessor`]):
            A image_processor to be used to preprocess image from clip.
    """

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: paddlenlp.transformers.CLIPVisionModelWithProjection,
        text_encoder: paddlenlp.transformers.CLIPTextModelWithProjection,
        tokenizer: paddlenlp.transformers.CLIPTokenizer,
        scheduler: UnCLIPScheduler,
        image_processor: paddlenlp.transformers.CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)
    def interpolate(
        self,
        images_and_prompts: List[Union[str, PIL.Image.Image, paddle.Tensor]],
        weights: List[float],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        negative_prior_prompt: Optional[str] = None,
        negative_prompt: str = "",
        guidance_scale: float = 4.0,
    ):
        """
        Function invoked when using the prior pipeline for interpolation.

        Args:
            images_and_prompts (`List[Union[str, PIL.Image.Image, paddle.Tensor:
                list of prompts and images to guide the image generation.
            weights: (`List[float]`):
                list of weights for each condition in `images_and_prompts`
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s).
                to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            negative_prior_prompt (`str`, *optional*):
                The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """
        if len(images_and_prompts) != len(weights):
            raise ValueError(
                f"`images_and_prompts` contains {len(images_and_prompts)} items and `weights` contains {len(weights)} items - they should be lists of same length"
            )
        image_embeddings = []
        for cond, weight in zip(images_and_prompts, weights):
            if isinstance(cond, str):
                image_emb = self(
                    cond,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    latents=latents,
                    negative_prompt=negative_prior_prompt,
                    guidance_scale=guidance_scale,
                ).image_embeds.unsqueeze(0)
            elif isinstance(cond, (PIL.Image.Image, paddle.Tensor)):
                if isinstance(cond, PIL.Image.Image):
                    cond = (
                        self.image_processor(cond, return_tensors="pd")
                        .pixel_values[0]
                        .unsqueeze(axis=0)
                        .cast(dtype=self.image_encoder.dtype)
                    )
                image_emb = (
                    self.image_encoder(cond)["image_embeds"]
                    .tile(repeat_times=[num_images_per_prompt, 1])
                    .unsqueeze(axis=0)
                )
            else:
                raise ValueError(
                    f"`images_and_prompts` can only contains elements to be of type `str`, `PIL.Image.Image` or `paddle.Tensor`  but is {type(cond)}"
                )
            image_embeddings.append(image_emb * weight)
        image_emb = paddle.concat(x=image_embeddings).sum(axis=0)
        out_zero = self(
            negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            negative_prompt=negative_prior_prompt,
            guidance_scale=guidance_scale,
        )
        zero_image_emb = out_zero.negative_image_embeds if negative_prompt == "" else out_zero.image_embeds
        return KandinskyPriorPipelineOutput(image_embeds=image_emb, negative_image_embeds=zero_image_emb)

    # Copied from ppdiffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents
        latents = latents * scheduler.init_noise_sigma
        return latents

    # Copied from ppdiffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline.get_zero_embed
    def get_zero_embed(self, batch_size=1):
        zero_img = paddle.zeros(
            shape=[1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size]
        ).cast(dtype=self.image_encoder.dtype)
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.tile(repeat_times=[batch_size, 1])
        return zero_image_emb

    # Copied from ppdiffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline._encode_prompt
    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids
        if (
            untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
            and not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item()
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                f"The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_encoder_output = self.text_encoder(text_input_ids)
        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state
        prompt_embeds = prompt_embeds.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(
            repeats=num_images_per_prompt, axis=0
        )
        text_mask = text_mask.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
                return_attention_mask=True,
            )
            uncond_text_mask = uncond_input.attention_mask
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids)
            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile(repeat_times=[1, num_images_per_prompt])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len])
            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.tile(
                repeat_times=[1, num_images_per_prompt, 1]
            )

            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.reshape(
                [batch_size * num_images_per_prompt, seq_len, -1]
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(repeats=num_images_per_prompt, axis=0)
            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = paddle.concat(
                x=[uncond_text_encoder_hidden_states, text_encoder_hidden_states]
            )
            text_mask = paddle.concat(x=[uncond_text_mask, text_mask])
        return prompt_embeds, text_encoder_hidden_states, text_mask

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        guidance_scale: float = 4.0,
        output_type: Optional[str] = "pd",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s).
                to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"`
                (`paddle.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif not isinstance(negative_prompt, list) and negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # if the negative prompt is defined we double the batch size to
        # directly retrieve the negative prompt embedding
        if negative_prompt is not None:
            prompt = prompt + negative_prompt
            negative_prompt = 2 * negative_prompt
        batch_size = len(prompt)
        batch_size = batch_size * num_images_per_prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # prior
        self.scheduler.set_timesteps(num_inference_steps)
        prior_timesteps_tensor = self.scheduler.timesteps
        embedding_dim = self.prior.config.embedding_dim
        latents = self.prepare_latents(
            (batch_size, embedding_dim), prompt_embeds.dtype, generator, latents, self.scheduler
        )
        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat(x=[latents] * 2) if do_classifier_free_guidance else latents
            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding
            if do_classifier_free_guidance:
                (predicted_image_embedding_uncond, predicted_image_embedding_text) = predicted_image_embedding.chunk(
                    chunks=2
                )
                predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )
            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]
            latents = self.scheduler.step(
                predicted_image_embedding, timestep=t, sample=latents, generator=generator, prev_timestep=prev_timestep
            ).prev_sample
        latents = self.prior.post_process_latents(latents)
        image_embeddings = latents

        # if negative prompt has been defined, we retrieve split the image embedding into two
        if negative_prompt is None:
            zero_embeds = self.get_zero_embed(latents.shape[0])

        else:
            image_embeddings, zero_embeds = image_embeddings.chunk(chunks=2)

        if output_type not in ["pd", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")
        if output_type == "np":
            image_embeddings = image_embeddings.cpu().numpy()
            zero_embeds = zero_embeds.cpu().numpy()
        if not return_dict:
            return image_embeddings, zero_embeds
        return KandinskyPriorPipelineOutput(image_embeds=image_embeddings, negative_image_embeds=zero_embeds)
