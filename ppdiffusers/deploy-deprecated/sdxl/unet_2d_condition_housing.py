# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from ppdiffusers import UNet2DConditionModel
from ppdiffusers.models.unet_2d_condition import UNet2DConditionOutput


class UNet2DConditionModelSDXLHousing(UNet2DConditionModel):
    def forward(
        self,
        sample: paddle.Tensor,
        timestep: Union[paddle.Tensor, float, int],
        encoder_hidden_states: paddle.Tensor,
        text_embeds: paddle.Tensor,
        time_ids: paddle.Tensor,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }

        return UNet2DConditionModel.forward(
            self,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
