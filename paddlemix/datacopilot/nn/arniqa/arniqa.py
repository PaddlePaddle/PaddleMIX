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
from __future__ import annotations

from pathlib import Path

import paddle
import paddle.nn.functional as F

from .pd_model_encoder.x2paddle_code import Sequential as encoder_paddle_model
from .pd_model_regressor.x2paddle_code import (
    TorchLinearRegression as regressor_paddle_model,
)


class ARNIQA(paddle.nn.Layer):
    """
    ARNIQA: Learning Distortion Manifold for Image Quality Assessment

    @inproceedings{agnolucci2024arniqa,
      title={ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
      author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={189--198},
      year={2024}
    }

    Reference:
        - Arxiv link: https://www.arxiv.org/abs/2310.14918
        - Official Github: https://github.com/miccunifi/ARNIQA
    """

    def __init__(
        self,
        default_mean: tuple[float] = (0.485, 0.456, 0.406),
        default_std: tuple[float] = (0.229, 0.224, 0.225),
        feat_dim: int = 2048,
    ):
        super(ARNIQA, self).__init__()
        self.default_mean = paddle.to_tensor(default_mean).view([1, 3, 1, 1])
        self.default_std = paddle.to_tensor(default_std).view([1, 3, 1, 1])
        self.feat_dim = feat_dim
        self.encoder = encoder_paddle_model()
        self.regressor = regressor_paddle_model()

        encoder_paddle_params = paddle.load(str(Path(__file__).parent / "pd_model_encoder" / "model.pdparams"))
        regressor_paddle_params = paddle.load(str(Path(__file__).parent / "pd_model_regressor" / "model.pdparams"))

        self.encoder.set_dict(encoder_paddle_params, use_structured_name=True)
        self.regressor.set_dict(regressor_paddle_params, use_structured_name=True)

    def forward(self, x: paddle.Tensor) -> float:
        x, x_ds = self._preprocess(x)

        f = F.normalize(self.encoder(x), axis=1)
        f_ds = F.normalize(self.encoder(x_ds), axis=1)
        f_combined = paddle.hstack((f, f_ds)).reshape([-1, self.feat_dim * 2])

        score = self.regressor(f_combined)
        score = self._scale_score(score)

        return score

    def _preprocess(self, x: paddle.Tensor):
        x_ds = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = (x - self.default_mean) / self.default_std
        x_ds = (x_ds - self.default_mean) / self.default_std
        return x, x_ds

    def _scale_score(self, score: float) -> float:
        new_range = (0.0, 1.0)

        # Compute scaling factors
        original_range = (1, 100)
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        return scaled_score

    def __call__(self, item: paddle.Tensor) -> float:
        return self.forward(item)

    def inference(self, item: paddle.Tensor) -> float:
        return self.forward(item)
