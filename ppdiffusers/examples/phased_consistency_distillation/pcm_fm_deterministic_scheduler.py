# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import paddle

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.utils import BaseOutput, logging
from ppdiffusers.utils.paddle_utils import randn_tensor
from ppdiffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class PCMFMDeterministicSchedulerOutput(BaseOutput):
    prev_sample: paddle.Tensor


class PCMFMDeterministicScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        pcm_timesteps: int = 50,
    ):
        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        timesteps = paddle.to_tensor(timesteps).to(dtype=paddle.float32)
        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.euler_timesteps = (
            np.arange(1, pcm_timesteps + 1) * (num_train_timesteps // pcm_timesteps)
        ).round().astype(np.int64) - 1
        self.sigmas = sigmas.numpy()[::-1][self.euler_timesteps]
        self.sigmas = paddle.to_tensor((self.sigmas[::-1].copy()))
        self.timesteps = self.sigmas * num_train_timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
        noise: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Forward process in flow-matching

        Args:
            sample (`paddle.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `paddle.Tensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self, num_inference_steps: int,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps
        inference_indices = np.linspace(
            0, self.config.pcm_timesteps, num=num_inference_steps, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = paddle.to_tensor(inference_indices).astype("int64")

        self.sigmas_ = self.sigmas[inference_indices]
        timesteps = self.sigmas_ * self.config.num_train_timesteps
        self.timesteps = timesteps
        self.sigmas_ = paddle.concat(x=[self.sigmas_, paddle.zeros(shape=[1])])
        print(self.sigmas_)

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
    ) -> Union[PCMFMDeterministicSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`paddle.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or (isinstance(timestep, paddle.Tensor) and "int" in str(timestep.dtype)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(paddle.float32)

        sigma = self.sigmas_[self.step_index]

        denoised = sample - model_output * sigma
        derivative = (sample - denoised) / sigma

        dt = self.sigmas_[self.step_index + 1] - sigma
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return PCMFMDeterministicSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
