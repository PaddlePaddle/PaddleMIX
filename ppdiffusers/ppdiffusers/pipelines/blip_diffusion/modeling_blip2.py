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
from paddlenlp.transformers.activations import QuickGELUActivation as QuickGELU
from paddlenlp.transformers.blip_2.configuration import Blip2Config, Blip2VisionConfig
from paddlenlp.transformers.blip_2.modeling import (
    Blip2Encoder,
    Blip2QFormerAttention,
    Blip2QFormerIntermediate,
    Blip2QFormerOutput,
)
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from paddlenlp.transformers.model_utils import apply_chunking_to_forward

from ppdiffusers.transformers import BertTokenizer, PretrainedModel

from ...utils import logging

logger = logging.get_logger(__name__)


class Blip2PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Blip2Config
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
    ]
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2D) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=factor)
            if hasattr(module, "padding_idx") and module.padding_idx is not None:
                module.weight[module.padding_idx] = 0.0
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, Blip2VisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            trunc_normal_ = nn.initializer.TruncatedNormal(mean=0.0, std=factor)
            trunc_normal_(module.position_embedding)
            trunc_normal_(
                module.class_embedding,
            )
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


# There is an implementation of Blip2 in `transformers` : https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py.
# But it doesn't support getting multimodal embeddings. So, this module can be
# replaced with a future `transformers` version supports that.
class Blip2TextEmbeddings(nn.Layer):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )  # padding_idx=config.pad_token_id  NOTE, do not set padding_idx
        self.word_embeddings.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype=paddle.int64).expand((1, -1))
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.shape[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                batch_size = embeddings.shape[0]
                # repeat the query embeddings for batch size
                query_embeds = query_embeds.tile([batch_size, 1, 1])
                embeddings = paddle.concat((query_embeds, embeddings), axis=1)
        else:
            embeddings = query_embeds
        embeddings = embeddings.cast(query_embeds.dtype)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copy-pasted from transformers.models.blip.modeling_blip.BlipVisionEmbeddings with Blip->Blip2
class Blip2VisionEmbeddings(nn.Layer):
    def __init__(self, config: Blip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(paddle.randn([1, 1, self.embed_dim]))

        self.patch_embedding = nn.Conv2D(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias_attr=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(paddle.randn([1, self.num_positions, self.embed_dim]))

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.cast(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose([0, 2, 1])

        class_embeds = self.class_embedding.expand([batch_size, 1, -1]).cast(target_dtype)
        embeddings = paddle.concat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.shape[1], :].cast(target_dtype)
        return embeddings


# The Qformer encoder, which takes the visual embeddings, and the text input, to get multimodal embeddings
class Blip2QFormerEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        if getattr(self.config, "gradient_checkpointing", False) and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and not hidden_states.stop_gradient:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# The layers making up the Qformer encoder
class Blip2QFormerLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(config)

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = Blip2QFormerIntermediate(config)
        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)
        self.output = Blip2QFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = paddle.concat([layer_output, layer_output_text], axis=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


# ProjLayer used to project the multimodal Blip2 embeddings to be used in the text encoder
class ProjLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, epsilon=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x


# Copy-pasted from transformers.models.blip.modeling_blip.BlipVisionModel with Blip->Blip2, BLIP->BLIP_2
class Blip2VisionModel(Blip2PretrainedModel):
    main_input_name = "pixel_values"
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = Blip2VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)
        self.encoder = Blip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
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

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


# Qformer model, used to get multimodal embeddings from the text and image inputs
class Blip2QFormerModel(Blip2PretrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.config = config
        self.embeddings = Blip2TextEmbeddings(config.qformer_config)
        self.visual_encoder = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(paddle.zeros([1, config.num_query_tokens, config.qformer_config.hidden_size]))
        if not hasattr(config, "tokenizer") or config.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer, truncation_side="right")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.proj_layer = ProjLayer(
            in_dim=config.qformer_config.hidden_size,
            out_dim=config.qformer_config.hidden_size,
            hidden_dim=config.qformer_config.hidden_size * 4,
            drop_p=0.1,
            eps=1e-12,
        )

        self.encoder = Blip2QFormerEncoder(config.qformer_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_extended_attention_mask(
        self,
        attention_mask: paddle.Tensor,
        input_shape: Tuple[int],
        has_query: bool = False,
    ) -> paddle.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.cast(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        text_input=None,
        image_input=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(paddle.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        text = self.tokenizer(text_input, return_tensors="pd", padding=True, return_attention_mask=True)
        input_ids = text.input_ids
        batch_size = input_ids.shape[0]
        query_atts = paddle.ones((batch_size, self.query_tokens.shape[1]), dtype=paddle.int64)
        attention_mask = paddle.concat([query_atts, text.attention_mask], axis=1)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = self.query_tokens.shape[1]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            query_embeds=self.query_tokens,
            past_key_values_length=past_key_values_length,
        )

        # embedding_output = self.layernorm(query_embeds)
        # embedding_output = self.dropout(embedding_output)

        input_shape = embedding_output.shape[:-1]
        batch_size, seq_length = input_shape

        image_embeds_frozen = self.visual_encoder(image_input).last_hidden_state
        # image_embeds_frozen = paddle.ones_like(image_embeds_frozen)
        encoder_hidden_states = image_embeds_frozen

        if attention_mask is None:
            attention_mask = paddle.ones(
                ((batch_size, seq_length + past_key_values_length)),
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].shape
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return self.proj_layer(sequence_output[:, :query_length, :])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
