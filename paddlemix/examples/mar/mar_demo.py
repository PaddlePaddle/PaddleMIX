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

import os

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
print(parent_path)
import sys

sys.path.append(parent_path)
import argparse
import random

import numpy as np
import paddle
from IPython.display import display
from models.mar import mar
from models.mar.APIs.self_use_save_image import save_image
from models.mar.vae import AutoencoderKL
from PIL import Image

paddle.set_device("gpu:0")

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="mar_huge", help="type of the model")
parser.add_argument(
    "--model_path", type=str, default="exchange/mar/paddle_mar_huge.pdparams", help="path of the model"
)
parser.add_argument("--vae_path", type=str, default="exchange/mar/VAE_kl16_ckpt.pdparams", help="path of the vae")
parser.add_argument(
    "--labels",
    type=str,
    default="[207, 360, 388, 113, 355, 980, 323, 979]",
    help="labels for the images that need to be generated",
)
args = parser.parse_args()

model_type = args.model_type  # @param ["mar_base", "mar_large", "mar_huge"]
model_path = args.model_path
vae_path = args.vae_path
labels = eval(args.labels)

num_sampling_steps_diffloss = 100  # @param {type:"slider", min:1, max:1000, step:1}
if model_type == "mar_base":
    diffloss_d = 6
    diffloss_w = 1024
elif model_type == "mar_large":
    diffloss_d = 8
    diffloss_w = 1280
elif model_type == "mar_huge":
    diffloss_d = 12
    diffloss_w = 1536
else:
    raise NotImplementedError

model = mar.__dict__[model_type](
    buffer_size=64, diffloss_d=diffloss_d, diffloss_w=diffloss_w, num_sampling_steps=str(num_sampling_steps_diffloss)
)

state_dict = paddle.load(model_path)
model.set_state_dict(state_dict)
model.eval()  # important!

vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_path)
vae.eval()
# Set user inputs:
seed = 0  # @param {type:"number"}
random.seed(seed)
paddle.seed(seed)
np.random.seed(seed)
num_ar_steps = 64  # @param {type:"slider", min:1, max:256, step:1}
cfg_scale = 4  # @param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant"  # @param ["linear", "constant"]
temperature = 1.0  # @param {type:"slider", min:0.9, max:1.1, step:0.01}
class_labels = labels  # @param {type:"list"}
samples_per_row = 4  # @param {type:"number"}
labels = paddle.to_tensor(class_labels, dtype="int64").cuda()

# Set user inputs:
seed = 0  # @param {type:"number"}
random.seed(seed)
paddle.seed(seed)
np.random.seed(seed)
num_ar_steps = 64  # @param {type:"slider", min:1, max:256, step:1}
cfg_scale = 4  # @param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant"  # @param ["linear", "constant"]
temperature = 1.0  # @param {type:"slider", min:0.9, max:1.1, step:0.01}
class_labels = [207, 360, 388, 113, 355, 980, 323, 979]  # @param {type:"raw"}
samples_per_row = 4  # @param {type:"number"}

labels = paddle.to_tensor(class_labels, dtype="int64").cuda()

with paddle.no_grad():  # important!
    with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level="O1"):
        sampled_tokens = model.sample_tokens(
            bsz=len(class_labels),
            num_iter=num_ar_steps,
            cfg=cfg_scale,
            cfg_schedule=cfg_schedule,
            labels=labels,
            temperature=temperature,
            progress=True,
        )
        print("mean", paddle.mean(sampled_tokens))
        print("min", paddle.min(sampled_tokens))
        print("max", paddle.max(sampled_tokens))
        sampled_images = vae.decode(sampled_tokens / 0.2325)

save_image(sampled_images, "sample.png", nrow=int(samples_per_row), normalize=True, value_range=(-1, 1))
samples = Image.open("sample.png")
display(samples)
