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


import paddle
import paddle.nn.functional as F

from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2LMHead,
)


class PPDocBee2TransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    layer_idx = 15

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
        Returns:
            `paddle.Tensor`: hidden_states.
        """
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `paddle.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = paddle.to_tensor(data=cu_window_seqlens, dtype="int32", place=hidden_states.place)
        cu_window_seqlens = paddle.unique_consecutive(x=cu_window_seqlens)
        seq_len, _ = tuple(hidden_states.shape)
        hidden_states = hidden_states.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        multi_vit = []
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.enable_recompute and self.training:
                hidden_states = self.recompute_training_full(blk, hidden_states, cu_seqlens_now, rotary_pos_emb)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)

            multi_vit.append(hidden_states)
        layer_idx = type(self).layer_idx
        hidden_states = self.merger(hidden_states + multi_vit[layer_idx])
        reverse_indices = paddle.argsort(x=window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class PPDocBee2ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, attn_implementation="flash_attention_2"):
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = attn_implementation

        self.visual = PPDocBee2TransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(config, embedding_weights=self.model.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        self.enable_recompute = False
