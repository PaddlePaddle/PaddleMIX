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

from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(paddle.float64)

    # calculation
    sinusoid = paddle.outer(position, paddle.pow(10000, -paddle.arange(half).to(position).div(half)))
    x = paddle.cat([paddle.cos(sinusoid), paddle.sin(sinusoid)], dim=1)
    return x


def Teacache_forward(
    self,
    hidden_states: paddle.Tensor,
    timestep: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    encoder_hidden_states_image: Optional[paddle.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
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

    # with amp.autocast(dtype=torch.float32):
    #     e = self.time_embedding(
    #         sinusoidal_embedding_1d(self.freq_dim, t).float())
    #     e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    #     assert e.dtype == torch.float32 and e0.dtype == torch.float32
    e = temb.to(paddle.float32)
    e0 = timestep_proj.to(paddle.float32)
    assert e.dtype == paddle.float32 and e0.dtype == paddle.float32
    if self.enable_teacache:
        modulated_inp = e0 if self.use_ref_steps else e
        # teacache
        if self.cnt % 2 == 0:  # even -> conditon
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean())
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else:  # odd -> unconditon
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean())
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    # 4. Transformer blocks
    if self.enable_teacache:
        if self.is_even:
            if not should_calc_even:
                hidden_states += self.previous_residual_even
            else:
                ori_x = hidden_states.clone()
                if paddle.is_grad_enabled() and self.gradient_checkpointing:
                    for block in self.blocks:
                        hidden_states = self._gradient_checkpointing_func(
                            block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                        )
                    self.previous_residual_even = hidden_states - ori_x
                else:
                    for block in self.blocks:
                        hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                    self.previous_residual_even = hidden_states - ori_x
        else:
            if not should_calc_odd:
                hidden_states += self.previous_residual_odd
            else:
                ori_x = hidden_states.clone()
                if paddle.is_grad_enabled() and self.gradient_checkpointing:
                    for block in self.blocks:
                        hidden_states = self._gradient_checkpointing_func(
                            block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                        )
                    self.previous_residual_odd = hidden_states - ori_x
                else:
                    for block in self.blocks:
                        hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                    self.previous_residual_odd = hidden_states - ori_x
    else:
        if paddle.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
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

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
