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

from ppdiffusers import DPMSolverSDEScheduler
from ppdiffusers.utils import paddle_device
from ppdiffusers.utils.testing_utils import require_paddlesde

from .test_schedulers import SchedulerCommonTest


@require_paddlesde
class DPMSolverSDESchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverSDEScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "noise_sampler_seed": 0,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        if paddle_device in ["mps"]:
            assert abs(result_sum.item() - 167.47821044921875) < 1e-2
            assert abs(result_mean.item() - 0.2178705964565277) < 1e-3
        elif paddle_device in ["gpu"]:
            assert abs(result_sum.item() - 171.59352111816406) < 1e-2
            assert abs(result_mean.item() - 0.22342906892299652) < 1e-3
        else:
            assert abs(result_sum.item() - 162.52383422851562) < 1e-2
            assert abs(result_mean.item() - 0.211619570851326) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        if paddle_device in ["mps"]:
            assert abs(result_sum.item() - 124.77149200439453) < 1e-2
            assert abs(result_mean.item() - 0.16226289014816284) < 1e-3
        elif paddle_device in ["gpu"]:
            assert abs(result_sum.item() - 128.1663360595703) < 1e-2
            assert abs(result_mean.item() - 0.16688326001167297) < 1e-3
        else:
            assert abs(result_sum.item() - 119.8487548828125) < 1e-2
            assert abs(result_mean.item() - 0.1560530662536621) < 1e-3

    def test_full_loop_device(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=paddle_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        if paddle_device in ["mps"]:
            assert abs(result_sum.item() - 167.46957397460938) < 1e-2
            assert abs(result_mean.item() - 0.21805934607982635) < 1e-3
        elif paddle_device in ["gpu"]:
            assert abs(result_sum.item() - 171.59353637695312) < 1e-2
            assert abs(result_mean.item() - 0.22342908382415771) < 1e-3
        else:
            assert abs(result_sum.item() - 162.52383422851562) < 1e-2
            assert abs(result_mean.item() - 0.211619570851326) < 1e-3

    def test_full_loop_device_karras_sigmas(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config, use_karras_sigmas=True)

        scheduler.set_timesteps(self.num_inference_steps, device=paddle_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        if paddle_device in ["mps"]:
            assert abs(result_sum.item() - 176.66974135742188) < 1e-2
            assert abs(result_mean.item() - 0.23003872730981811) < 1e-2
        elif paddle_device in ["gpu"]:
            assert abs(result_sum.item() - 177.63653564453125) < 1e-2
            assert abs(result_mean.item() - 0.23003872730981811) < 1e-2
        else:
            assert abs(result_sum.item() - 170.3135223388672) < 1e-2
            assert abs(result_mean.item() - 0.23003872730981811) < 1e-2
