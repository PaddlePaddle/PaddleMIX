# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/ocr/sd3.5-M"
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(
        8 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14  # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/geneval/sd3.5-M"
    config.reward_fn = {
        "geneval": 1.0,
    }

    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/pickscore/sd3.5-M"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def clipscore_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.02
    config.sample.global_std = True
    config.sample.same_latent = True
    config.train.ema = True
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/clipscore/sd3.5-M"
    config.reward_fn = {
        "clipscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_s1():
    # gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 1
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 16  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-4
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 5
    config.train.ema = True
    config.save_freq = 30  # epoch
    config.eval_freq = 30
    config.save_dir = "logs/pickscore/sd3.5-M-s1"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_4gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(
        16 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/ocr/sd3.5-M"
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_4gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(
        32 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/pickscore/sd3.5-M"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_1gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(
        64 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60  # epoch
    config.eval_freq = 60
    config.save_dir = "logs/ocr/sd3.5-M"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_flux():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 30  # epoch
    config.eval_freq = 30
    config.save_dir = "logs/pickscore/flux-group24"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_flux_8gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 9
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 30  # epoch
    config.eval_freq = 30
    config.save_dir = "logs/pickscore/flux-group24-8gpu"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def counting_flux_kontext():
    gpu_number = 28
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # sd3.5 medium
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 21
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert (
        config.sample.num_batches_per_epoch % 2 == 0
    ), "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 30  # epoch
    config.eval_freq = 30
    config.save_dir = "logs/counting_edit/flux_kontext"
    config.reward_fn = {
        "image_similarity": 0.5,
        "geneval": 0.5,
    }
    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
