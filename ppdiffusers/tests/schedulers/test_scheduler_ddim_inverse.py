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

import paddle
from ppdiffusers import DDIMInverseScheduler
from .test_schedulers import SchedulerCommonTest


class DDIMInverseSchedulerTest(SchedulerCommonTest):
    scheduler_classes = DDIMInverseScheduler,
    forward_default_kwargs = ('eta', 0.0), ('num_inference_steps', 50)

    def get_scheduler_config(self, **kwargs):
        config = {'num_train_timesteps': 1000, 'beta_start': 0.0001,
            'beta_end': 0.02, 'beta_schedule': 'linear', 'clip_sample': True}
        config.update(**kwargs)
        return config

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        num_inference_steps, eta = 10, 0.0
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, eta).prev_sample
        return sample

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(5)
        assert paddle.equal_all(x=scheduler.timesteps, y=paddle.to_tensor(
            data=[-199, 1, 201, 401, 601], dtype='int64')).item()

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002,
            0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ['linear', 'squaredcos_cap_v2']:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ['epsilon', 'v_prediction']:
            self.check_over_configs(prediction_type=prediction_type)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_timestep_spacing(self):
        for timestep_spacing in ['trailing', 'leading']:
            self.check_over_configs(timestep_spacing=timestep_spacing)

    def test_rescale_betas_zero_snr(self):
        for rescale_betas_zero_snr in [True, False]:
            self.check_over_configs(rescale_betas_zero_snr=
                rescale_betas_zero_snr)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ['epsilon', 'v_prediction']:
                self.check_over_configs(thresholding=True, prediction_type=
                    prediction_type, sample_max_value=threshold)

    def test_time_indices(self):
        for t in [1, 10, 49]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
            self.check_over_forward(time_step=t, num_inference_steps=
                num_inference_steps)

    def test_add_noise_device(self):
        pass

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_sum = paddle.sum(x=paddle.abs(x=sample))
        result_mean = paddle.mean(x=paddle.abs(x=sample))
        assert abs(result_sum.item() - 509.1079) < 0.01
        assert abs(result_mean.item() - 0.6629) < 0.001

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type='v_prediction')
        result_sum = paddle.sum(x=paddle.abs(x=sample))
        result_mean = paddle.mean(x=paddle.abs(x=sample))
        assert abs(result_sum.item() - 1029.129) < 0.01
        assert abs(result_mean.item() - 1.34) < 0.001

    def test_full_loop_with_set_alpha_to_one(self):
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = paddle.sum(x=paddle.abs(x=sample))
        result_mean = paddle.mean(x=paddle.abs(x=sample))
        assert abs(result_sum.item() - 259.8116) < 0.01
        assert abs(result_mean.item() - 0.3383) < 0.001

    def test_full_loop_with_no_set_alpha_to_one(self):
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = paddle.sum(x=paddle.abs(x=sample))
        result_mean = paddle.mean(x=paddle.abs(x=sample))
        assert abs(result_sum.item() - 239.055) < 0.01
        assert abs(result_mean.item() - 0.3113) < 0.001
