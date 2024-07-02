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

from typing import Optional  # noqa

import paddle
from paddle import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin


class StableUnCLIPImageNormalizer(ModelMixin, ConfigMixin):
    """
    This class is used to hold the mean and standard deviation of the CLIP embedder used in stable unCLIP.

    It is used to normalize the image embeddings before the noise is applied and un-normalize the noised image
    embeddings.
    """

    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 768,
    ):
        super().__init__()

        self.mean = nn.Parameter(paddle.zeros([1, embedding_dim]))
        self.std = nn.Parameter(paddle.ones([1, embedding_dim]))

    # def to(
    #     self,
    #     paddle_device: Optional[str] = None,
    #     paddle_dtype: Optional[paddle.dtype] = None,
    # ):
    #     self.mean._to(device=paddle_device, dtype=paddle_dtype)
    #     self.std._to(device=paddle_device, dtype=paddle_dtype)
    #     return self

    def scale(self, embeds):
        embeds = (embeds - self.mean) * 1.0 / self.std
        return embeds

    def unscale(self, embeds):
        embeds = (embeds * self.std) + self.mean
        return embeds
