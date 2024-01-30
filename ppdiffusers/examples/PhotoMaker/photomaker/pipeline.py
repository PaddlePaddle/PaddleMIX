import paddle
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import PIL
from ppdiffusers import StableDiffusionXLPipeline
from ppdiffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from ppdiffusers.transformers.clip import CLIPImageProcessor

from . import PhotoMakerIDEncoder
from . import rescale_noise_cfg

device = paddle.device.get_device()

PipelineImageInput = Union[PIL.Image.Image, paddle.Tensor,
                           List[PIL.Image.Image], List[paddle.Tensor]]

class PhotoMakerStableDiffusionXLPipeline(StableDiffusionXLPipeline):

    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str,
                                                               paddle.Tensor]],
        weight_name: str,
        subfolder: str = "",
        trigger_word: str = "img",
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A paddle state dict.

            weight_name (`str`):
                The weight name NOT the path to the weight.

            subfolder (`str`, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

            trigger_word (`str`, *optional*, defaults to `"img"`):
                The trigger word is used to identify the position of class word in the text prompt, 
                and it is recommended not to set it as a common word. 
                This trigger word must be placed after the class word when used, otherwise, it will affect the performance of the personalized generation.           
        """

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = paddle.load(pretrained_model_name_or_path_or_dict)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError(
                "Required keys are (`id_encoder` and `lora_weights`) missing from the state dict."
            )
        self.trigger_word = trigger_word


        id_encoder = PhotoMakerIDEncoder()
        id_encoder.set_state_dict(state_dict=state_dict["id_encoder"],
                                  use_structured_name=True)

        self.id_encoder = id_encoder.to(dtype=self.unet.dtype)

        self.id_image_processor = CLIPImageProcessor()


        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker", from_diffusers=True)

        if self.tokenizer is not None:
            self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_id_images: int = 1,
        prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        class_tokens_mask: Optional[paddle.Tensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        image_token_id = self.tokenizer_2.convert_tokens_to_ids(
            self.trigger_word)

        tokenizers = ([self.tokenizer, self.tokenizer_2]
                      if self.tokenizer is not None else [self.tokenizer_2])
        text_encoders = ([
            self.text_encoder, self.text_encoder_2
        ] if self.text_encoder is not None else [self.text_encoder_2])

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]

            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers,
                                                       text_encoders):
                input_ids = tokenizer.encode(prompt)["input_ids"]
                clean_index = 0
                clean_input_ids = []
                class_token_index = []

                for i, token_id in enumerate(input_ids):
                    if token_id == image_token_id:
                        class_token_index.append(clean_index - 1)
                    else:
                        clean_input_ids.append(token_id)
                        clean_index += 1

                if len(class_token_index) != 1:
                    raise ValueError(
                        f"PhotoMaker currently does not support multiple trigger words in a single prompt.Trigger word: {self.trigger_word}, Prompt: {prompt}."
                    )

                class_token_index = class_token_index[0]
                class_token = clean_input_ids[class_token_index]
                clean_input_ids = (clean_input_ids[:class_token_index] +
                                   [class_token] * num_id_images +
                                   clean_input_ids[class_token_index + 1:])

                max_len = tokenizer.model_max_length

                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                else:
                    clean_input_ids = clean_input_ids + [
                        tokenizer.pad_token_id
                    ] * (max_len - len(clean_input_ids))

                class_tokens_mask = [
                    (True if class_token_index <= i < class_token_index +
                     num_id_images else False)
                    for i in range(len(clean_input_ids))
                ]

                clean_input_ids = paddle.to_tensor(
                    data=clean_input_ids, dtype="int64").unsqueeze(axis=0)
                class_tokens_mask = paddle.to_tensor(
                    data=class_tokens_mask, dtype="bool").unsqueeze(axis=0)
                prompt_embeds = text_encoder(clean_input_ids,
                                             output_hidden_states=True)
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)
            prompt_embeds = paddle.concat(x=prompt_embeds_list, axis=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype)
        class_tokens_mask = class_tokens_mask

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator,
                                  List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: int = 1,
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 0,
        class_tokens_mask: Optional[paddle.Tensor] = None,
        prompt_embeds_text_only: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds_text_only: Optional[paddle.Tensor] = None,
    ):
        """
        Function invoked when calling the pipeline for generation.
        Only the parameters introduced by PhotoMaker are discussed here.
        For explanations of the previous parameters in StableDiffusionXLPipeline, please refer to https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py

        Args:
            input_id_images (`PipelineImageInput`, *optional*):
                Input ID Image to work with PhotoMaker.
            class_tokens_mask (`paddle.Tensor`, *optional*):
                Pre-generated class token. When the `prompt_embeds` parameter is provided in advance, it is necessary to prepare the `class_tokens_mask` beforehand for marking out the position of class word.
            prompt_embeds_text_only (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds_text_only (`paddle.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        if prompt_embeds is not None and class_tokens_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `class_tokens_mask` also have to be passed. Make sure to generate `class_tokens_mask` from the same tokenizer that was used to generate `prompt_embeds`."
            )
        if input_id_images is None:
            raise ValueError(
                "Provide `input_id_images`. Cannot leave `input_id_images` undefined for PhotoMaker pipeline."
            )

        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale >= 1.0
        assert do_classifier_free_guidance

        num_id_images = len(input_id_images)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            class_tokens_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt=prompt,
            prompt_2=prompt_2,
            num_id_images=num_id_images,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            class_tokens_mask=class_tokens_mask,
        )

        tokens_text_only = self.tokenizer.encode(prompt,
                                                 add_special_tokens=False)
        trigger_word_token = self.tokenizer.convert_tokens_to_ids(
            self.trigger_word)

        tokens_text_only["input_ids"].remove(trigger_word_token)
        prompt_text_only = self.tokenizer.decode(tokens_text_only["input_ids"],
                                                 add_special_tokens=False)

        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        if not isinstance(input_id_images[0], paddle.Tensor):
            id_pixel_values = self.id_image_processor(
                input_id_images, return_tensors="pd").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(axis=0)
        self.id_encoder.to(device)
        prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds,
                                        class_tokens_mask)
        bs_embed, seq_len, _ = prompt_embeds.shape

        # release graphics memory
        self.id_encoder.to("cpu")
        paddle.device.cuda.empty_cache()
        
        prompt_embeds = prompt_embeds.cast(dtype=self.id_encoder.dtype)
        prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])

        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])
        pooled_prompt_embeds = pooled_prompt_embeds.cast(dtype=self.id_encoder.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.tile(repeat_times=[1, num_images_per_prompt]).reshape(
                [bs_embed * num_images_per_prompt, -1])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = paddle.concat(x=[add_time_ids, add_time_ids], axis=0)
        add_time_ids = add_time_ids.tile(repeat_times=[batch_size * num_images_per_prompt,1])

        # release graphics memory
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        paddle.device.cuda.empty_cache()
        
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        
        self.unet.to(device)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (paddle.concat(
                    x=[latents] *
                    2) if do_classifier_free_guidance else latents)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                if i <= start_merge_step:
                    current_prompt_embeds = paddle.concat(
                        x=[negative_prompt_embeds, prompt_embeds_text_only],
                        axis=0)
                    add_text_embeds = paddle.concat(
                        x=[negative_pooled_prompt_embeds,pooled_prompt_embeds_text_only],
                        axis=0,
                    )
                else:
                    current_prompt_embeds = paddle.concat(
                        x=[negative_prompt_embeds, prompt_embeds], 
                        axis=0)
                    add_text_embeds = paddle.concat(
                        x=[negative_pooled_prompt_embeds, pooled_prompt_embeds], 
                        axis=0)
                
                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=guidance_rescale)
                latents = self.scheduler.step(noise_pred,
                                              t,
                                              latents,
                                              **extra_step_kwargs,
                                              return_dict=False)[0]
                if (i == len(timesteps) - 1 or i + 1 > num_warmup_steps and
                        (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # release graphics memory
        self.unet.to("cpu")
        paddle.device.cuda.empty_cache()

        if self.vae.dtype == "float16" and self.vae.config.force_upcast:
            self.upcast_vae()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor,return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        # if self.watermark is not None:
        #     image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image,output_type=output_type)

        if not return_dict:
            return (image, )
        return StableDiffusionXLPipelineOutput(images=image)
