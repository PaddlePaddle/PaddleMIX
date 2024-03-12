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

# load related packages
import paddle
import torch

paddle.set_device("cpu")
from collections import OrderedDict

from safetensors import safe_open  # noqa: F401

if __name__ == "__main__":
    # torch model define
    ###
    from navit.main import NaViT

    torch_model = v = NaViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        token_dropout_prob=0.1,
    )
    ###

    # torch model data load
    in_torch_model_file_path = "navit.pth"
    out_paddle_model_file_path = "navit.pdparams"

    # load pt
    torch_model_data = torch.load(in_torch_model_file_path, map_location=torch.device("cpu"))
    # load safetensors
    # torch_model_data = {}
    # with safe_open(in_torch_model_file_path, framework="pt", device=0) as f:
    #     for k in f.keys():
    #         torch_model_data[k] = f.get_tensor(k)

    # torch model data to paddle model data
    need_transpose = []
    for k, v in torch_model.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")

    paddle_model_data = OrderedDict()
    for k, v in torch_model_data.items():
        if k not in need_transpose:
            paddle_model_data[k] = v.detach().cpu().numpy()
        else:
            paddle_model_data[k] = v.t().detach().cpu().numpy()

    # paddle model data save and check
    paddle.save(paddle_model_data, out_paddle_model_file_path)
    paddle_model_data = paddle.load(out_paddle_model_file_path)
    for p, t in zip(paddle_model_data.items(), torch_model_data.items()):
        print(p[0], "|", t[0])
