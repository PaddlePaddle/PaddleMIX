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

from .mpr_predictor import MPRPredictor as mpr  # 去模糊
from .nafnet_predictor import NAFNetPredictor as nafn  # 去模糊
from .realsr_predictor_df2k import RealSRPredictor as df2k  # 现有权重下，质量最好
from .realsr_predictor_drn import RealSRPredictor as drn  # 现有权重下，质量次于df2k，颜色鲜亮
from .realsr_predictor_esrgan import RealSRPredictor as esrgan  # 现有权重下，质量次于df2k，图片小一半
from .realsr_predictor_lesr import RealSRPredictor as lesr  # 现有权重下，质量次于df2k，吃显存不建议

__all__ = [
    df2k,
    esrgan,
    lesr,
    drn,
    mpr,
    nafn,
]
