# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import tempfile
from typing import Dict, List, Tuple

import paddle

from ppdiffusers import LCMScheduler

from .test_schedulers import SchedulerCommonTest


class LCMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LCMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.0120,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        }

        config.update(**kwargs)
        return config

    @property
    def default_valid_timestep(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)
        timestep = scheduler.timesteps[-1]
        return timestep

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            # 0 is not guaranteed to be in the timestep schedule, but timesteps - 1 is
            self.check_over_configs(time_step=timesteps - 1, num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(time_step=self.default_valid_timestep, beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear", "squaredcos_cap_v2"]:
            self.check_over_configs(time_step=self.default_valid_timestep, beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(time_step=self.default_valid_timestep, prediction_type=prediction_type)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(time_step=self.default_valid_timestep, clip_sample=clip_sample)

    def test_thresholding(self):
        self.check_over_configs(time_step=self.default_valid_timestep, thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "v_prediction"]:
                self.check_over_configs(
                    time_step=self.default_valid_timestep,
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_time_indices(self):
        # Get default timestep schedule.
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        for t in timesteps:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        # Hardcoded for now
        for t, num_inference_steps in zip([99, 39, 39, 19], [10, 25, 26, 50]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    # Override test_add_noise_device because the hardcoded num_inference_steps of 100 doesn't work
    # for LCMScheduler under default settings
    def test_add_noise_device(self, num_inference_steps=10):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            sample = self.dummy_sample
            scaled_sample = scheduler.scale_model_input(sample, 0.0)
            self.assertEqual(sample.shape, scaled_sample.shape)

            noise = paddle.randn_like(scaled_sample)
            t = scheduler.timesteps[5][None]
            noised = scheduler.add_noise(scaled_sample, noise, t)
            self.assertEqual(noised.shape, scaled_sample.shape)

    # Override test_from_save_pretrained because it hardcodes a timestep of 1
    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            timestep = self.default_valid_timestep

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            scheduler.set_timesteps(num_inference_steps)
            new_scheduler.set_timesteps(num_inference_steps)

            kwargs["generator"] = paddle.Generator().manual_seed(0)
            output = scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            kwargs["generator"] = paddle.Generator().manual_seed(0)
            new_output = new_scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    # Override test_step_shape because uses 0 and 1 as hardcoded timesteps
    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler.set_timesteps(num_inference_steps)

            timestep_0 = scheduler.timesteps[-2]
            timestep_1 = scheduler.timesteps[-1]

            output_0 = scheduler.step(residual, timestep_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, timestep_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    # Override test_set_scheduler_outputs_equivalence since it uses 0 as a hardcoded timestep
    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    paddle.allclose(
                        set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                    ),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {paddle.max(paddle.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {paddle.isnan(tuple_object).any()} and `inf`: {paddle.isinf(tuple_object)}. Dict has"
                        f" `nan`: {paddle.isnan(dict_object).any()} and `inf`: {paddle.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)

        timestep = self.default_valid_timestep

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler.set_timesteps(num_inference_steps)
            kwargs["generator"] = paddle.Generator().manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            scheduler.set_timesteps(num_inference_steps)
            kwargs["generator"] = paddle.Generator().manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    def full_loop(self, num_inference_steps=10, seed=0, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(seed)

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator).prev_sample

        return sample

    def test_full_loop_onestep(self):
        sample = self.full_loop(num_inference_steps=1)

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        # TODO: get expected sum and mean
        assert abs(result_sum.item() - 18.7097) < 1e-3
        assert abs(result_mean.item() - 0.0244) < 1e-3

    def test_full_loop_multistep(self):
        sample = self.full_loop(num_inference_steps=10)

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        # TODO: get expected sum and mean
        assert abs(result_sum.item() - 197.1515) < 1e-3
        assert abs(result_mean.item() - 0.2567) < 1e-3

    def test_custom_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]

        scheduler.set_timesteps(timesteps=timesteps)

        scheduler_timesteps = scheduler.timesteps

        for i, timestep in enumerate(scheduler_timesteps):
            if i == len(timesteps) - 1:
                expected_prev_t = -1
            else:
                expected_prev_t = timesteps[i + 1]

            prev_t = scheduler.previous_timestep(timestep)
            prev_t = prev_t.item()

            self.assertEqual(prev_t, expected_prev_t)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 51, 0]

        with self.assertRaises(ValueError, msg="`custom_timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `custom_timesteps`."):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    def test_custom_timesteps_too_large(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [scheduler.config.num_train_timesteps]

        with self.assertRaises(
            ValueError,
            msg="`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}",
        ):
            scheduler.set_timesteps(timesteps=timesteps)
