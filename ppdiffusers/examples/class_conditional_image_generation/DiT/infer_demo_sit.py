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

import json
import random

import numpy as np
import paddle
from PIL import Image
from transport import Sampler, create_transport
from transport.sit import SiT

from ppdiffusers import AutoencoderKL

paddle.device.set_device("gpu")

image_size = "256"
vae_model = "stabilityai/sd-vae-ft-ema"  # will be downloaded automatically
config_file = "config/SiT_XL_patch2.json"


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


model = SiT(**read_json(config_file))
state_dict = "SiT-XL-2-256x256.pdparams"  # wget https://bj.bcebos.com/v1/paddlenlp/models/community/facebook/SiT-XL-2-256x256.pdparams
model.set_state_dict(paddle.load(state_dict))
model.eval()  # important!
vae = AutoencoderKL.from_pretrained(vae_model)
latent_size = 256 // 8


# Set user inputs:
seed = 0  # @param {type:"number"}
paddle.seed(seed)
random.seed(seed)
np.random.seed(seed)
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

tracker = get_rng_state_tracker()
tracker.add("global_seed", seed)

num_sampling_steps = 25  # @param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 4  # @param {type:"slider", min:1, max:10, step:0.1}
class_labels = [207]  # @param {type:"raw"}
samples_per_row = 4  # @param {type:"number"}
sampler_type = "SDE"  # @param ["ODE", "SDE"]
# Note: ODE not support yet


# Create diffusion object:
transport = create_transport()
sampler = Sampler(transport)

# Create sampling noise:
n = len(class_labels)
z = paddle.randn([n, 4, latent_size, latent_size])
y = paddle.to_tensor(class_labels)

# Setup classifier-free guidance:
z = paddle.concat([z, z], 0)
y_null = paddle.to_tensor([1000] * n)
y = paddle.concat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
if sampler_type == "SDE":
    SDE_sampling_method = "Euler"  # @param ["Euler", "Heun"]
    diffusion_form = "linear"  # @param ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"]
    diffusion_norm = 1  # @param {type:"slider", min:0, max:10.0, step:0.1}
    last_step = "Mean"  # @param ["Mean", "Tweedie", "Euler"]
    last_step_size = 0.4  # @param {type:"slider", min:0, max:1.0, step:0.01}
    sample_fn = sampler.sample_sde(
        sampling_method=SDE_sampling_method,
        diffusion_form=diffusion_form,
        diffusion_norm=diffusion_norm,
        last_step_size=last_step_size,
        num_steps=num_sampling_steps,
    )
else:
    raise NotImplementedError("ODE not supported yet. Please use SDE instead.")

samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
samples = vae.decode(samples / 0.18215).sample

samples = (samples - samples.min()) / (samples.max() - samples.min()) * 255
npimg = samples.astype("uint8").numpy()[0]
npimg = np.transpose(npimg, (1, 2, 0))
img = Image.fromarray(npimg)
img.save("result_SiT_golden_retriever.png")
