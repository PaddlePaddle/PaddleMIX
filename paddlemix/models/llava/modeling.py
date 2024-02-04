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

import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
from paddle.distributed import fleet
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from paddlenlp.transformers.llama.modeling import ConcatSePMaskedLoss, LlamaLMHead
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast

from .base_model import LlavaMetaForCausalLM, LlavaMetaModel
from .configuration import LlavaConfig

__all__ = [
    "LlavaLlamaModel",
    "LlavaLlamaForCausalLM",
]


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.llama = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = LlamaLMHead(config)
        self.criterion = LlavaCriterion(config)

    def get_model(self):
        return self.llama

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
                input_ids, position_ids, attention_mask, past_key_values, labels, images
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs

    # def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
    #     is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
    #     is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
    #         (eos_token_id is not None) and (pad_token_id != eos_token_id)
    #     )
    #     if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
    #         attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * get_scale_by_dtype(
    #             return_positive=False
    #         )
    #     else:
    #         attention_mask = paddle.ones_like(input_ids, dtype=paddle.get_default_dtype())
    #     return attention_mask


class LlavaCriterion(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(self, config):

        super(LlavaCriterion, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        with paddle.amp.auto_cast(False):

            masked_lm_loss = self.loss_func(
                prediction_scores[..., :-1, :].astype("float32"), masked_lm_labels.unsqueeze(2)[..., 1:, :]
            )

            if self.config.sep_parallel_degree > 1:
                _hcg = fleet.get_hybrid_communicate_group()
                masked_lm_loss = ConcatSePMaskedLoss.apply(masked_lm_loss, axis=1, group=_hcg.get_sep_parallel_group())
            # skip ignore_index which loss == 0
            masked_lm_loss = masked_lm_loss[masked_lm_loss > 0].astype("float32")
            loss = paddle.mean(masked_lm_loss)

        return loss
