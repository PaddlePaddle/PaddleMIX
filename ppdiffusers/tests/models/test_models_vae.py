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

import gc
import unittest

import paddle
from parameterized import parameterized

from ppdiffusers import AsymmetricAutoencoderKL, AutoencoderKL
from ppdiffusers.utils import (
    floats_tensor,
    load_hf_numpy,
    paddle_all_close,
    require_paddle_gpu,
    slow,
)
from ppdiffusers.utils.testing_utils import enable_full_determinism

from .test_modeling_common import ModelTesterMixin, UNetTesterMixin

enable_full_determinism()


class AutoencoderKLTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL
    main_input_name = "sample"
    base_precision = 1e-2

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = 32, 32
        image = floats_tensor((batch_size, num_channels) + sizes)
        return {"sample": image}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_determinism(self):
        super().test_determinism(expected_max_diff=1e-4)

    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.clear_gradients()

        labels = paddle.randn(out.shape, dtype=out.dtype)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_dict(model.state_dict())
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.clear_gradients()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-5)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        with paddle.no_grad():
            for name, param in named_params.items():
                self.assertTrue(paddle_all_close(param.grad, named_params_2[name].grad, atol=5e-5))

    def test_from_pretrained_hub(self):
        model, loading_info = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)
        image = model(**self.dummy_input)
        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy")
        model.eval()

        generator = paddle.Generator().manual_seed(0)
        image = paddle.randn(
            shape=[1, model.config.in_channels, model.config.sample_size, model.config.sample_size],
            generator=paddle.Generator().manual_seed(0),
        )
        with paddle.no_grad():
            output = model(image, sample_posterior=True, generator=generator).sample
        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        # Since the VAE Gaussian prior's generator is seeded on the appropriate device,
        # the expected output slices are not the same for CPU and GPU.
        expected_output_slice = paddle.to_tensor(
            [
                -0.39049336,
                0.34836933,
                0.27105471,
                -0.02148458,
                0.00975929,
                0.27822807,
                -0.12224892,
                -0.02011922,
                0.19761699,
            ]
        )
        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=0.01))


class AsymmetricAutoencoderKLTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AsymmetricAutoencoderKL
    main_input_name = "sample"
    base_precision = 0.01

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = 32, 32
        image = floats_tensor((batch_size, num_channels) + sizes)
        mask = paddle.ones(shape=(batch_size, 1) + sizes)
        return {"sample": image, "mask": mask}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "down_block_out_channels": [32, 64],
            "layers_per_down_block": 1,
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "up_block_out_channels": [32, 64],
            "layers_per_up_block": 1,
            "act_fn": "silu",
            "latent_channels": 4,
            "norm_num_groups": 32,
            "sample_size": 32,
            "scaling_factor": 0.18215,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_forward_with_norm_groups(self):
        pass


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = paddle.float16 if fp16 else paddle.float32
        image = paddle.to_tensor(data=load_hf_numpy(self.get_file_format(seed, shape))).cast(dtype)
        return image

    def get_sd_vae_model(self, model_id="CompVis/stable-diffusion-v1-4", fp16=False):
        revision = "fp16" if fp16 else None
        paddle_dtype = paddle.float16 if fp16 else paddle.float32
        model = AutoencoderKL.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle_dtype, revision=revision)
        return model

    def get_generator(self, seed=0):
        return paddle.Generator().manual_seed(seed)

    @parameterized.expand(
        [
            [
                33,
                [-0.1603, 0.9878, -0.0495, -0.079, -0.2709, 0.8375, -0.206, -0.0824],
            ],
            [
                47,
                [-0.2376, 0.1168, 0.1332, -0.484, -0.2508, -0.0791, -0.0493, -0.4089],
            ],
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)
        with paddle.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample
        assert sample.shape == image.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32").cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [33, [-0.0513, 0.0289, 1.3799, 0.2166, -0.2573, -0.0871, 0.5103, -0.0999]],
            [47, [-0.4128, -0.132, -0.3704, 0.1965, -0.4116, -0.2332, -0.334, 0.2247]],
        ]
    )
    @require_paddle_gpu
    def test_stable_diffusion_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        image = self.get_sd_image(seed, fp16=True)
        generator = self.get_generator(seed)
        with paddle.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample
        assert sample.shape == image.shape
        output_slice = sample[-1, -2:, :2, -2:].flatten().cast("float32").cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [
                33,
                [-0.1609, 0.9866, -0.0487, -0.0777, -0.2716, 0.8368, -0.2055, -0.0814],
            ],
            [
                47,
                [-0.2377, 0.1147, 0.1333, -0.4841, -0.2506, -0.0805, -0.0491, -0.4085],
            ],
        ]
    )
    def test_stable_diffusion_mode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        with paddle.no_grad():
            sample = model(image).sample
        assert sample.shape == image.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32").cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [13, [-0.2051, -0.1803, -0.2311, -0.2114, -0.3292, -0.3574, -0.2953, -0.3323]],
            [37, [-0.2632, -0.2625, -0.2199, -0.2741, -0.4539, -0.499, -0.372, -0.4925]],
        ]
    )
    @require_paddle_gpu
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))
        with paddle.no_grad():
            sample = model.decode(encoding).sample
        assert list(sample.shape) == [3, 3, 512, 512]
        output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [27, [-0.0369, 0.0207, -0.0776, -0.0682, -0.1747, -0.193, -0.1465, -0.2039]],
            [16, [-0.1628, -0.2134, -0.2747, -0.2642, -0.3774, -0.4404, -0.3687, -0.4277]],
        ]
    )
    @require_paddle_gpu
    def test_stable_diffusion_decode_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)
        with paddle.no_grad():
            sample = model.decode(encoding).sample
        assert list(sample.shape) == [3, 3, 512, 512]
        output_slice = sample[-1, -2:, :2, -2:].flatten().cast("float32").cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.005)

    @parameterized.expand([(13,), (16,), (27,)])
    @require_paddle_gpu
    def test_stable_diffusion_decode_ppxformers_vs_2_5_fp16(self, seed):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)

        with paddle.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with paddle.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert paddle_all_close(sample, sample_2, atol=1e-1)

    @parameterized.expand([(13,), (16,), (37,)])
    @require_paddle_gpu
    def test_stable_diffusion_decode_ppxformers_vs_2_5(self, seed):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with paddle.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with paddle.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert paddle_all_close(sample, sample_2, atol=1e-2)

    @parameterized.expand(
        [
            [33, [-0.3001, 0.0918, -2.6984, -3.972, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.503, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)
        with paddle.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)
        assert list(sample.shape) == [image.shape[0], 4] + [(i // 8) for i in image.shape[2:]]
        output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        tolerance = 0.01
        assert paddle_all_close(output_slice, expected_output_slice, atol=tolerance)

    def test_stable_diffusion_model_local(self):
        model_id = "stabilityai/sd-vae-ft-mse"
        model_1 = AutoencoderKL.from_pretrained(model_id)
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        model_2 = AutoencoderKL.from_single_file(url)
        image = self.get_sd_image(33)
        with paddle.no_grad():
            sample_1 = model_1(image).sample
            sample_2 = model_2(image).sample
        assert sample_1.shape == sample_2.shape
        output_slice_1 = sample_1[-1, -2:, -2:, :2].flatten().astype(dtype="float32").cpu()
        output_slice_2 = sample_2[-1, -2:, -2:, :2].flatten().astype(dtype="float32").cpu()
        assert paddle_all_close(output_slice_1, output_slice_2, atol=0.003)


@slow
class AsymmetricAutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = "float16" if fp16 else "float32"
        image = paddle.to_tensor(data=load_hf_numpy(self.get_file_format(seed, shape))).cast(dtype)
        return image

    def get_sd_vae_model(self, model_id="cross-attention/asymmetric-autoencoder-kl-x-1-5", fp16=False):
        revision = "main"
        paddle_dtype = "float32"
        model = AsymmetricAutoencoderKL.from_pretrained(model_id, paddle_dtype=paddle_dtype, revision=revision)
        model.eval()
        return model

    def get_generator(self, seed=0):
        return paddle.Generator().manual_seed(seed)

    @parameterized.expand(
        [
            [
                33,
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
            ],
            [
                47,
                [0.44, 0.0543, 0.2873, 0.2946, 0.0553, 0.0839, -0.1585, 0.2529],
            ],
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)
        with paddle.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample
        assert sample.shape == image.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().astype(dtype="float32").cpu()
        expected_output_slice = paddle.to_tensor(data=expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.005)

    @parameterized.expand(
        [
            [
                33,
                [-0.034, 0.287, 0.1698, -0.0105, -0.3448, 0.3529, -0.1321, 0.1097],
            ],
            [
                47,
                [0.4397, 0.055, 0.2873, 0.2946, 0.0567, 0.0855, -0.158, 0.2531],
            ],
        ]
    )
    def test_stable_diffusion_mode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        with paddle.no_grad():
            sample = model(image).sample
        assert sample.shape == image.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().astype(dtype="float32").cpu()
        expected_output_slice = paddle.to_tensor(data=expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.003)

    @parameterized.expand(
        [
            [13, [-0.0521, -0.2939, 0.154, -0.1855, -0.5936, -0.3138, -0.4579, -0.2275]],
            [37, [-0.182, -0.4345, -0.0455, -0.2923, -0.8035, -0.5089, -0.4795, -0.3106]],
        ]
    )
    @require_paddle_gpu
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))
        with paddle.no_grad():
            sample = model.decode(encoding).sample
        assert list(sample.shape) == [3, 3, 512, 512]
        output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
        expected_output_slice = paddle.to_tensor(data=expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.002)

    @parameterized.expand(
        [
            [33, [-0.3001, 0.0918, -2.6984, -3.972, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.503, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)
        with paddle.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)
        assert list(sample.shape) == [image.shape[0], 4] + [(i // 8) for i in image.shape[2:]]
        output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = paddle.to_tensor(data=expected_slice)
        tolerance = 0.003
        assert paddle_all_close(output_slice, expected_output_slice, atol=tolerance)
