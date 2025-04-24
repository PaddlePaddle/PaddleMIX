from types import MethodType
from .tgate_utils import register_forward,tgate_scheduler
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import PIL.Image
import paddle
from ppdiffusers.utils import replace_example_docstring
from ppdiffusers.utils.paddle_utils import randn_tensor

from ppdiffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims,
    retrieve_timesteps,
    StableVideoDiffusionPipelineOutput
    )

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


@paddle.no_grad()
def tgate(
    self,
    image: Union[PIL.Image.Image, List[PIL.Image.Image], paddle.Tensor],
    height: int = 576,
    width: int = 1024,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 25,
    sigmas: Optional[List[float]] = None,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: Optional[int] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
    latents: Optional[paddle.Tensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    return_dict: bool = True,
    gate_step: int = 10,
    sp_interval: int = 5,
    fi_interval: int = 1,
    warm_up: int = 2,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `paddle.Tensor`):
            Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
            1]`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        num_frames (`int`, *optional*):
            The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
            `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
        num_inference_steps (`int`, *optional*, defaults to 25):
            The number of denoising steps. More denoising steps usually lead to a higher quality video at the
            expense of slower inference. This parameter is modulated by `strength`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        min_guidance_scale (`float`, *optional*, defaults to 1.0):
            The minimum guidance scale. Used for the classifier free guidance with first frame.
        max_guidance_scale (`float`, *optional*, defaults to 3.0):
            The maximum guidance scale. Used for the classifier free guidance with last frame.
        fps (`int`, *optional*, defaults to 7):
            Frames per second. The rate at which the generated images shall be exported to a video after
            generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
        motion_bucket_id (`int`, *optional*, defaults to 127):
            Used for conditioning the amount of motion for the generation. The higher the number the more motion
            will be in the video.
        noise_aug_strength (`float`, *optional*, defaults to 0.02):
            The amount of noise added to the init image, the higher it is the less the video will look like the
            init image. Increase it for more motion.
        decode_chunk_size (`int`, *optional*):
            The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
            expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
            For lower memory usage, reduce `decode_chunk_size`.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
            A `paddle.Generator` to make
            generation deterministic.
        latents (`paddle.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `pil`, `np` or `pt`.
        callback_on_step_end (`Callable`, *optional*):
            A function that is called at the end of each denoising step during inference. The function is called
            with the following arguments:
                `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
            `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        gate_step (`int` defaults to 10): The time step to stop calculating the cross attention.
        sp_interval (`int` defaults to 5): The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).
        fi_interval (`int` defaults to 1): The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).
        warm_up (`int` defaults to 2): The time step to warm up the model inference.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
            returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `paddle.Tensor`)
            is returned.
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
    decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(image, height, width)

    # 2. Define call parameters
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    self._guidance_scale = max_guidance_scale

    # 3. Encode input image
    image_embeddings = self._encode_image(image, num_videos_per_prompt, self.do_classifier_free_guidance)
    negative_image_embeddings = self._encode_image(image, num_videos_per_prompt, False)

    # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
    # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
    fps = fps - 1

    # 4. Encode input image using VAE
    image = self.video_processor.preprocess(image, height=height, width=width)
    noise = randn_tensor(image.shape, generator=generator, dtype=image.dtype)
    image = image + noise_aug_strength * noise

    needs_upcasting = self.vae.dtype == paddle.float16 and self.vae.config.force_upcast
    if needs_upcasting:
        self.vae.to(dtype=paddle.float32)

    image_latents = self._encode_vae_image(
        image,
        num_videos_per_prompt=num_videos_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
    )
    image_latents = image_latents.to(image_embeddings.dtype)

    negative_image_latents = self._encode_vae_image(
        image,
        num_videos_per_prompt=num_videos_per_prompt,
        do_classifier_free_guidance=False,
    )
    negative_image_latents = negative_image_latents.to(image_embeddings.dtype)

    # cast back to fp16 if needed
    if needs_upcasting:
        self.vae.to(dtype=paddle.float16)

    # Repeat the image latents for each frame so we can concatenate them with the noise
    # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
    image_latents = image_latents.unsqueeze(1).tile([1, num_frames, 1, 1, 1])
    negative_image_latents = negative_image_latents.unsqueeze(1).tile([1, num_frames, 1, 1, 1])

    # 5. Get Added Time IDs
    added_time_ids = self._get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        image_embeddings.dtype,
        batch_size,
        num_videos_per_prompt,
        self.do_classifier_free_guidance,
    )

    negative_added_time_ids = self._get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        image_embeddings.dtype,
        batch_size,
        num_videos_per_prompt,
        False,
    )
    
    # 6. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, None, sigmas)

    # 7. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_frames,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        generator,
        latents,
    )

    # 8. Prepare guidance scale
    guidance_scale = paddle.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
    guidance_scale = guidance_scale.cast(latents.dtype)
    guidance_scale = guidance_scale.tile([batch_size * num_videos_per_prompt, 1])
    guidance_scale = _append_dims(guidance_scale, latents.ndim)

    self._guidance_scale = guidance_scale

    # 9. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    register_forward(self.unet, 
        'Attention',
        ca_kward = {
            'cache': False,
            'reuse': False,
        },
        sa_kward = {
            'cache': False,
            'reuse': False,
        },
        keep_shape=True
        )

    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if self.do_classifier_free_guidance and (i-num_warmup_steps)<gate_step: 
                latent_model_input = paddle.concat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            else:
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                image_latents = negative_image_latents
                image_latents = negative_image_latents
                added_time_ids=negative_added_time_ids
                image_embeddings = negative_image_embeddings
            # Concatenate image_latents over channels dimension
            latent_model_input = paddle.concat([latent_model_input, image_latents], axis=2)


            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):

                ca_kwards,sa_kwards,keep_shape=tgate_scheduler(
                    cur_step=i-num_warmup_steps, 
                    gate_step=gate_step,
                    sp_interval=sp_interval,
                    fi_interval=fi_interval,
                    warm_up=warm_up
                )
                register_forward(self.unet, 
                    ca_kward=ca_kwards,
                    sa_kward=sa_kwards,
                    keep_shape=keep_shape
                    )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance and (i-num_warmup_steps)<gate_step:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    if not output_type == "latent":
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=paddle.float16)
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
    else:
        frames = latents

    self.maybe_free_model_hooks()

    if not return_dict:
        return frames

    return StableVideoDiffusionPipelineOutput(frames=frames)



def TgateSVDLoader(pipe):
    pipe.tgate = MethodType(tgate,pipe)
    return pipe