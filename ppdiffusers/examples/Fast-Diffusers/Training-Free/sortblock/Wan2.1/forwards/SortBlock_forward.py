# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
from cache_functions import cache_init
from taylorseer_utils import derivative_approximation, taylor_formula

from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)


def SortBlock_forward(
    self,
    hidden_states: paddle.Tensor,
    timestep: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    encoder_hidden_states_image: Optional[paddle.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
    if attention_kwargs is None:
        attention_kwargs = {}
    if attention_kwargs.get("cache_dic", None) is None:
        attention_kwargs["cache_dic"], attention_kwargs["current"] = cache_init(self)
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = paddle.concat([encoder_hidden_states_image, encoder_hidden_states], axis=1)

    self.count += 1

    if timestep == 999:
        self.current_single_block_residual = [None] * len(self.blocks)
        self.previous_single_block_residual = [None] * len(self.blocks)
        self.count = 0
        self.percentage = 1
        self.result_list = []
        self.result_single_list = []
    is_within_block_range = self.end <= timestep <= self.start
    if is_within_block_range:
        self.step_Num = self.step_Num2
    if timestep < self.end or timestep > self.start:
        self.step_Num = 1

    # 4. Transformer blocks
    attention_kwargs["current"]["stream"] = "single_stream"
    cache_dic = attention_kwargs["cache_dic"]
    current = attention_kwargs["current"]
    if self.count % self.step_Num == 0:
        current["activated_steps"].append(current["step"])
    if paddle.is_grad_enabled() and self.gradient_checkpointing:
        for index_block, block in enumerate(self.blocks):
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for index_block, block in enumerate(self.blocks):
            attention_kwargs["current"]["layer"] = index_block
            should_compute_block = self.count % self.step_Num == 0 or (
                self.result_single_list != [] and self.result_single_list[index_block] == 1
            )
            if should_compute_block:
                ori_hidden_states = hidden_states.clone()
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                current["module"] = "hidden_states"
                derivative_approximation(
                    cache_dic=cache_dic, current=current, feature=hidden_states.clone() - ori_hidden_states
                )
                if self.count % self.step_Num == 0:
                    self.previous_single_block_residual[index_block] = hidden_states.clone() - ori_hidden_states
            else:
                if self.count % self.step_Num == 1:
                    current["module"] = "hidden_states"
                    self.current_single_block_residual[index_block] = taylor_formula(
                        cache_dic=cache_dic, current=current
                    )
                current["module"] = "hidden_states"
                hidden_states += taylor_formula(cache_dic=cache_dic, current=current)

    if self.count % self.step_Num == 1:
        coefficients = [-1.74367e-15, 1.20871e-11, -2.11429e-8, 1.49777e-5, -0.00453, 0.4971]
        rescale_func = np.poly1d(coefficients)
        self.percentage = rescale_func(timestep.item()) * self.beta
        cosine_single_similarities = []
        for i in range(len(self.blocks)):
            cosine_similarity = paddle.nn.functional.cosine_similarity(
                self.previous_single_block_residual[i][
                    :, :, : self.previous_single_block_residual[i].shape[-1] // 8
                ].to(paddle.float32),
                self.current_single_block_residual[i][:, :, : self.current_single_block_residual[i].shape[-1] // 8].to(
                    paddle.float32
                ),
                axis=-1,
            )
            cosine_single_similarities.append(cosine_similarity.mean().item())
        sorted_cos = sorted(cosine_single_similarities)
        threshold = sorted_cos[int(len(self.blocks) * self.percentage)]
        self.result_single_list = []
        for j in cosine_single_similarities:
            if j <= threshold:
                self.result_single_list.append(1)
            else:
                self.result_single_list.append(0)

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, axis=1)
    hidden_states = (self.norm_out(hidden_states.cast(paddle.float32)) * (1 + scale) + shift).cast(hidden_states.dtype)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        [batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1]
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    attention_kwargs["current"]["step"] += 1
    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
