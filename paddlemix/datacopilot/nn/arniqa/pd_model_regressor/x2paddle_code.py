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

import paddle


class TorchLinearRegression(paddle.nn.Layer):
    def __init__(self):
        super(TorchLinearRegression, self).__init__()
        self.linear_weights = paddle.nn.Linear(in_features=4096, out_features=1)

    def forward(self, x0):
        x1 = 1
        x6 = self.linear_weights(x0)
        x7 = x6.shape
        x8 = x7[x1]
        x9 = x8 == x1
        if x9:
            x12 = paddle.reshape(x=x6, shape=[-1])
            x10 = x12
        else:
            x10 = x6
        return x10


def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 2048], type-float32.
    paddle.disable_static()
    params = paddle.load(r"/workspace/X2Paddle/test_benchmark/PyTorch/ARNIQA/pd_model_regressor/model.pdparams")
    model = TorchLinearRegression()
    model.set_dict(params, use_structured_name=True)
    model.eval()
    out = model(x0)
    return out
