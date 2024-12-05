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

from typing import Tuple, Union

import paddle

__all__ = ["MlpProjector"]


class MlpProjector(paddle.nn.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["projector_type"] == "identity":
            modules = paddle.nn.Identity()
        elif cfg["projector_type"] == "linear":
            modules = paddle.nn.Linear(in_features=cfg["input_dim"], out_features=cfg["n_embed"])
        elif cfg["projector_type"] == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [paddle.nn.Linear(in_features=cfg["input_dim"], out_features=cfg["n_embed"])]
            for _ in range(1, mlp_depth):
                modules.append(paddle.nn.GELU())
                modules.append(paddle.nn.Linear(in_features=cfg["n_embed"], out_features=cfg["n_embed"]))
            modules = paddle.nn.Sequential(*modules)
        elif cfg["projector_type"] == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = paddle.nn.Linear(in_features=cfg["input_dim"], out_features=cfg["n_embed"] // 2)
            self.low_up_proj = paddle.nn.Linear(in_features=cfg["input_dim"], out_features=cfg["n_embed"] // 2)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(paddle.nn.GELU())
                modules.append(paddle.nn.Linear(in_features=cfg["n_embed"], out_features=cfg["n_embed"]))
            modules = paddle.nn.Sequential(*modules)
        else:
            raise ValueError(f'Unknown projector type: {cfg["projector_type"]}')
        self.layers = modules

    def forward(self, x_or_tuple: Union[Tuple[paddle.Tensor, paddle.Tensor], paddle.Tensor]):
        """

        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """
        if isinstance(x_or_tuple, tuple):
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = paddle.concat(x=[high_x, low_x], axis=-1)
        else:
            x = x_or_tuple
        return self.layers(x)
