# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# William Peebles and Saining Xie
#
# Copyright (c) 2021 OpenAI
# MIT License
#
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

from typing import Dict, List, Optional, Tuple, Union

import paddle

from ...models import AutoencoderKL, Transformer2DModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils.paddle_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`Transformer2DModel`]):
            A class conditioned `Transformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: Transformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)
        self.id2label = id2label
        # create a imagenet -> id dictionary for easier use
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]

    @paddle.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`paddle.Generator`, *optional*):
                A [`paddle.Generator`] to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from ppdiffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import paddle

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=paddle.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = paddle.Generator().manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            dtype=self.transformer.dtype,
        )
        latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = paddle.to_tensor(class_labels).reshape(
            [
                -1,
            ]
        )
        class_null = paddle.to_tensor([1000] * batch_size)
        class_labels_input = paddle.concat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = paddle.concat([half, half], axis=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t

            if not paddle.is_tensor(timesteps):
                if isinstance(timesteps, float):
                    dtype = paddle.float32
                else:
                    dtype = paddle.int64
                timesteps = paddle.to_tensor([timesteps], dtype=dtype)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None]
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(
                [
                    latent_model_input.shape[0],
                ]
            )
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, class_labels=class_labels_input
            ).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = paddle.chunk(eps, 2, axis=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = paddle.concat([half_eps, half_eps], axis=0)

                noise_pred = paddle.concat([eps, rest], axis=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = paddle.split(
                    noise_pred, [latent_channels, noise_pred.shape[1] - latent_channels], axis=1
                )
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, axis=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clip(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.transpose([0, 2, 3, 1]).cast("float32").cpu().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
