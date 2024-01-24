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

from typing import List, Optional, Tuple, Union

import paddle

from ...models import UNet2DModel
from ...schedulers import ScoreSdeVeScheduler
from ...utils.paddle_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class ScoreSdeVePipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`ScoreSdeVeScheduler`]):
            A `ScoreSdeVeScheduler` to be used in combination with `unet` to denoise the encoded image.
    """

    unet: UNet2DModel
    scheduler: ScoreSdeVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: ScoreSdeVeScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @paddle.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`paddle.Generator`, `optional`):
                A [`paddle.Generator`] to make generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        sample = randn_tensor(shape, generator=generator) * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * paddle.ones(
                [
                    shape[0],
                ]
            )

            # correction step
            for _ in range(self.scheduler.config.correct_steps):
                model_output = self.unet(sample, sigma_t).sample
                sample = self.scheduler.step_correct(model_output, sample, generator=generator).prev_sample

            # prediction step
            model_output = model(sample, sigma_t).sample
            output = self.scheduler.step_pred(model_output, t, sample, generator=generator)

            sample, sample_mean = output.prev_sample, output.prev_sample_mean

        sample = sample_mean.clip(0, 1)
        sample = sample.transpose([0, 2, 3, 1]).cast("float32").cpu().numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)
