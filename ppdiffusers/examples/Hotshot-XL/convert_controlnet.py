# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
from collections import OrderedDict

import paddle
import torch

paddle.set_device("cpu")


def convert_to_ppdiffusers(vae_or_unet, dtype=None):
    need_transpose = []
    for k, v in vae_or_unet.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = OrderedDict()
    for k, v in vae_or_unet.state_dict().items():
        if k not in need_transpose:
            new_vae_or_unet[k] = v.cpu().numpy()
            if dtype is not None:
                new_vae_or_unet[k] = new_vae_or_unet[k].astype(dtype)
        else:
            new_vae_or_unet[k] = v.t().cpu().numpy()
            if dtype is not None:
                new_vae_or_unet[k] = new_vae_or_unet[k].astype(dtype)
    return new_vae_or_unet


from diffusers import ControlNetModel as TorchControlNetModel

from ppdiffusers import ControlNetModel


def convert_diffusers_stable_diffusion_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    model = TorchControlNetModel.from_pretrained(pretrained_model_name_or_path)
    state_dict = convert_to_ppdiffusers(model)
    model_new = ControlNetModel.from_config(model.config)
    model_new.set_state_dict(state_dict)
    model_new.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="diffusers/controlnet-depth-sdxl-1.0",
        # default="diffusers/controlnet-canny-sdxl-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="controlnet_depth",
        # default="controlnet_canny",
        help="The model output path.",
    )
    args = parser.parse_args()
    ppdiffusers_pipe = convert_diffusers_stable_diffusion_to_ppdiffusers(args.pretrained_path, args.output_path)
