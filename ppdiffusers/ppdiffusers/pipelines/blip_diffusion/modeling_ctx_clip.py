# Copyright 2023 Salesforce.com, inc.
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
from typing import Optional, Tuple, Union

import paddle
from paddle import nn
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPooling

from ppdiffusers.transformers import CLIPPretrainedModel
from ppdiffusers.transformers.clip.configuration import CLIPTextConfig
from ppdiffusers.transformers.clip.modeling import CLIPEncoder


def _expand_mask(mask: paddle.Tensor, dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)

    inverted_mask = 1.0 - expanded_mask

    return paddle.masked_fill(inverted_mask, inverted_mask.cast(paddle.bool), paddle.finfo(dtype).min)


# This is a modified version of the CLIPTextModel from transformers.models.clip.modeling_clip
# Which allows for an extra input of "context embeddings", which are the query embeddings used in Qformer
# They pass through the clip model, along with the text embeddings, and interact with them using self attention
class ContextCLIPTextModel(CLIPPretrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = ContextCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        ctx_embeddings: paddle.Tensor = None,
        ctx_begin_pos: list = None,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.text_model(
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ContextCLIPTextTransformer(nn.Layer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = ContextCLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        ctx_embeddings: paddle.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.reshape([-1, input_shape[-1]])

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
        )

        bsz, seq_len = input_shape
        if ctx_embeddings is not None:
            seq_len += ctx_embeddings.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            bsz,
            seq_len,
            hidden_states.dtype,
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to paddle.int32 for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state.gather_nd(
                paddle.stack(
                    [paddle.arange(last_hidden_state.shape[0], dtype="int32"), input_ids.argmax(-1, dtype="int32")],
                    axis=-1,
                )
            )
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of extra new tokens is possible)
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            pooled_output = last_hidden_state.gather_nd(
                paddle.stack(
                    [
                        paddle.arange(last_hidden_state.shape[0], dtype="int32"),
                        (input_ids == paddle.to_tensor([self.eos_token_id]))
                        .cast("int32")
                        .argmax(axis=-1, dtype="int32"),
                    ],
                    axis=-1,
                )
            )

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = paddle.triu(
            # paddle.full((bsz, 1, seq_len, seq_len), paddle.finfo(dtype).min, dtype=dtype),
            paddle.ones((bsz, paddle.to_tensor([1]), seq_len, seq_len), dtype=dtype) * paddle.finfo(dtype).min,
            diagonal=1,
        )
        return mask


class ContextCLIPTextEmbeddings(nn.Layer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype=paddle.int64).expand((1, -1))
        )

    def forward(
        self,
        ctx_embeddings: paddle.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if ctx_embeddings is None:
            ctx_len = 0
        else:
            ctx_len = ctx_embeddings.shape[1]

        seq_length = (input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]) + ctx_len

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length].cast(paddle.int64)

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

            # for each input embeddings, add the ctx embeddings at the correct position
            input_embeds_ctx = []
            bsz = inputs_embeds.shape[0]

            if ctx_embeddings is not None:
                for i in range(bsz):
                    cbp = ctx_begin_pos[i]

                    prefix = inputs_embeds[i, :cbp]
                    # remove the special token embedding
                    suffix = inputs_embeds[i, cbp:]

                    input_embeds_ctx.append(paddle.concat([prefix, ctx_embeddings[i], suffix], axis=0))

                inputs_embeds = paddle.stack(input_embeds_ctx, axis=0)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
