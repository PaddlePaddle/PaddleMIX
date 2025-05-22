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

from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddlenlp.transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

__all__ = [
    "LlavaQwenConfig",
    "LlavaQwenModel",
    "LlavaQwenForCausalLM",
]


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"
    mm_patch_merge_type = "spatial_unpad"
    use_cachekv_int8 = None


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class Qwen2LMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None, transpose_y=False, tensor_parallel_output=1):
        super(Qwen2LMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            # only for weight from embedding_weights
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            # for weight from model init
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )

    def forward(self, hidden_states, tensor_parallel_output=1):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=self.transpose_y)
        return logits


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig
    base_model_prefix = "llava_qwen"

    def __init__(self, config):
        super().__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        self.qwen2 = LlavaQwenModel(config)

        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(config, embedding_weights=self.qwen2.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)

    def get_model(self):
        return self.qwen2

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.Tensor] = None,
        image_size: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, image_size
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @paddle.no_grad()
    def generate(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = paddle.ones(shape=inputs_embeds.shape[:2], dtype="int64")

            batch_size, seq_length = attention_mask.shape
            position_ids = paddle.arange(seq_length).expand((batch_size, seq_length))

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
