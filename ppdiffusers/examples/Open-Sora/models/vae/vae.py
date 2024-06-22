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
from einops import rearrange

from ppdiffusers.models import AutoencoderKL


class VideoAutoencoderKL(paddle.nn.Layer):
    def __init__(self, from_pretrained=None, micro_batch_size=None, cache_dir=None, local_files_only=False):
        super().__init__()

        self.module = AutoencoderKL.from_pretrained(
            from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only, paddle_dtypes=paddle.float16
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().multiply(y=paddle.to_tensor(0.18215, dtype="float32"))
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = (
                    self.module.encode(x_bs)
                    .latent_dist.sample()
                    .multiply(y=paddle.to_tensor(0.18215, dtype="float32"))
                )
                x_out.append(x_bs)
            x = paddle.concat(x=x_out, axis=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
                x_out.append(x_bs)
            x = paddle.concat(x=x_out, axis=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
