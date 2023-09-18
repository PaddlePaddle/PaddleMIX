# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import paddle

from ...configuration_utils import FrozenDict
from ...models import LVDMAutoencoderKL, LVDMUNet3DModel
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import deprecate, logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from . import VideoPipelineOutput
from .video_save import save_results

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LVDMUncondPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vae ([`LVDMAutoencoderKL`]):
            Autoencoder Model to encode and decode videos to and from latent representations.
        unet ([`LVDMUNet3DModel`]): 3D conditional U-Net architecture to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: LVDMAutoencoderKL,
        unet: LVDMUNet3DModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
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

        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

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

    @paddle.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_frames: Optional[int] = 16,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        eta: Optional[float] = 0.0,
        num_inference_steps: Optional[int] = 50,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        save_dir=None,
        save_name=None,
        scale_factor: Optional[float] = 0.33422927,
        shift_factor: Optional[float] = 1.4606637,
        **kwargs,
    ) -> Union[Tuple, VideoPipelineOutput]:
        r"""
        Args:
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.VideoPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            save_dir (`str` or `List[str]`, *optional*):
                If provided, will save videos generated to *save_dir*. Otherwise will save them to the current path.
            save_name (`str` or `List[str]`, *optional*):
                If provided, will save videos generated to *save_name*.
            scale_factor (`float`, *optional*, defaults to 0.33422927):
                A scale factor to apply to the generated video.
            shift_factor (`float`, *optional*, defaults to 1.4606637):
                A shift factor to apply to the generated video.

        Returns:
            [`~pipeline_utils.VideoPipelineOutput`] or `tuple`: [`~pipeline_utils.VideoPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get the initial random noise unless the user supplied it
        latents_shape = [
            batch_size,
            self.unet.in_channels,
            num_frames,
            height // 8,
            width // 8,
        ]  # (batch_size, C, N, H, W)

        if latents is None:
            latents = randn_tensor(
                latents_shape,
                generator=generator,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_tensor = paddle.expand(
                t,
                [
                    latent_model_input.shape[0],
                ],
            )
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t_tensor).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        all_videos = []
        latents = 1.0 / scale_factor * latents - shift_factor
        sampled_videos = self.vae.decode(latents).sample
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
