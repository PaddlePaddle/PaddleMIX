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

import sys

import paddle
from paddlemix.models.diffsinger.utils import paddle_aux

from paddlemix.models.diffsinger.utils import filter_kwargs

from .convnext import ConvNeXtDecoder

AUX_DECODERS = {"convnext": ConvNeXtDecoder}
AUX_LOSSES = {"convnext": paddle.nn.L1Loss}


def build_aux_decoder(in_dims: int, out_dims: int, aux_decoder_arch: str, aux_decoder_args: dict) -> paddle.nn.Layer:
    decoder_cls = AUX_DECODERS[aux_decoder_arch]
    kwargs = filter_kwargs(aux_decoder_args, decoder_cls)
    return AUX_DECODERS[aux_decoder_arch](in_dims, out_dims, **kwargs)


def build_aux_loss(aux_decoder_arch):
    return AUX_LOSSES[aux_decoder_arch]()


class AuxDecoderAdaptor(paddle.nn.Layer):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_feats: int,
        spec_min: list,
        spec_max: list,
        aux_decoder_arch: str,
        aux_decoder_args: dict,
    ):
        super().__init__()
        self.decoder = build_aux_decoder(
            in_dims=in_dims,
            out_dims=out_dims * num_feats,
            aux_decoder_arch=aux_decoder_arch,
            aux_decoder_args=aux_decoder_args,
        )
        self.out_dims = out_dims
        self.n_feats = num_feats
        if spec_min is not None and spec_max is not None:
            spec_min = paddle.to_tensor(data=spec_min, dtype="float32")[None, None, :].transpose(
                perm=paddle_aux.transpose_aux_func(
                    paddle.to_tensor(data=spec_min, dtype="float32")[None, None, :].ndim, -3, -2
                )
            )
            spec_max = paddle.to_tensor(data=spec_max, dtype="float32")[None, None, :].transpose(
                perm=paddle_aux.transpose_aux_func(
                    paddle.to_tensor(data=spec_max, dtype="float32")[None, None, :].ndim, -3, -2
                )
            )
            self.register_buffer(name="spec_min", tensor=spec_min, persistable=False)
            self.register_buffer(name="spec_max", tensor=spec_max, persistable=False)

    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.0
        b = (self.spec_max + self.spec_min) / 2.0
        return (x - b) / k

    def denorm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.0
        b = (self.spec_max + self.spec_min) / 2.0
        return x * k + b

    def forward(self, condition, infer=False):
        x = self.decoder(condition, infer=infer)
        if self.n_feats > 1:
            x = x.reshape(-1, tuple(x.shape)[1], self.n_feats, self.out_dims)
            x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        if infer:
            x = self.denorm_spec(x)
        return x
