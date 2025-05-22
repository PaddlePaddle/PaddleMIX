# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn

from ppdiffusers.transformers import GPT2Config, GPT2LMHeadModel
from ppdiffusers.transformers.model_utils import ModuleUtilsMixin

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...utils import logging

logger = logging.get_logger(__name__)


# Modified from ClipCaptionModel in https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py
class UniDiffuserTextDecoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    """
    Text decoder model for a image-text [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) model. This is used to
    generate text from the UniDiffuser image-text embedding.

    Parameters:
        prefix_length (`int`):
            Max number of prefix tokens that will be supplied to the model.
        prefix_inner_dim (`int`):
            The hidden size of the incoming prefix embeddings. For UniDiffuser, this would be the hidden dim of the
            CLIP text encoder.
        prefix_hidden_dim (`int`, *optional*):
            Hidden dim of the MLP if we encode the prefix.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
    """

    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.bias", r"h\.\d+\.attn\.masked_bias"]

    # TODO, use ppdiffusers.transformers.GPT2LMHeadModel
    _deprecated_dict = {
        "key": ".self_attn.q_proj.",
        "action": [{".attn.c_attn.": "concat"}],
        "name_mapping": {
            # common
            "gpt.": "transformer.",
            "decoder.layers.": "h.",
            "decoder.norm.": "ln_f.",
            # embeddings
            "embeddings.word_embeddings.": "wte.",
            "embeddings.position_embeddings.": "wpe.",
            # transformer
            ".self_attn.q_proj.": ".attn.c_attn.",  # need concat
            ".self_attn.k_proj.": ".attn.c_attn.",  # need concat
            ".self_attn.v_proj.": ".attn.c_attn.",  # need concat
            ".self_attn.out_proj.": ".attn.c_proj.",
            # other
            ".norm1.": ".ln_1.",
            ".linear1.": ".mlp.c_fc.",
            ".linear2.": ".mlp.c_proj.",
            ".norm2.": ".ln_2.",
        },
    }

    @classmethod
    def _update_deprecated_state_dict(cls, state_dict=None, loaded_keys=None, model=None):
        if state_dict is None:
            return loaded_keys
        _deprecated_dict = getattr(cls, "_deprecated_dict", None)
        from_deprecated_state_dict = _deprecated_dict is not None and any(
            cls._deprecated_dict.get("key", "NONE") in all_key for all_key in state_dict.keys()
        )
        if from_deprecated_state_dict:
            logger.warning(
                "Loading from deprecated state_dict, please load new state_dict via setting `use_safetensors=True`."
            )
            for name in list(state_dict.keys()):
                deprecated_name = name
                for old_name, new_name in cls._deprecated_dict.get("name_mapping", {}).items():
                    name = name.replace(old_name, new_name)

                if ".attn.c_attn." in name and name in state_dict:
                    state_dict[name] = paddle.concat([state_dict[name], state_dict.pop(deprecated_name)], axis=-1)
                else:
                    state_dict[name] = state_dict.pop(deprecated_name)
            loaded_keys = list(state_dict.keys())
        return loaded_keys

    @register_to_config
    def __init__(
        self,
        prefix_length: int,
        prefix_inner_dim: int,
        prefix_hidden_dim: Optional[int] = None,
        vocab_size: int = 50257,  # Start of GPT2 config args
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
    ):
        super().__init__()

        self.prefix_length = prefix_length

        if prefix_inner_dim != n_embd and prefix_hidden_dim is None:
            raise ValueError(
                f"`prefix_hidden_dim` cannot be `None` when `prefix_inner_dim`: {prefix_hidden_dim} and"
                f" `n_embd`: {n_embd} are not equal."
            )

        self.prefix_inner_dim = prefix_inner_dim
        self.prefix_hidden_dim = prefix_hidden_dim

        self.encode_prefix = (
            nn.Linear(self.prefix_inner_dim, self.prefix_hidden_dim)
            if self.prefix_hidden_dim is not None
            else nn.Identity()
        )
        self.decode_prefix = (
            nn.Linear(self.prefix_hidden_dim, n_embd) if self.prefix_hidden_dim is not None else nn.Identity()
        )
        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )
        self.transformer = GPT2LMHeadModel(gpt_config)

    def forward(
        self,
        input_ids: paddle.Tensor,
        prefix_embeds: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            input_ids (`paddle.Tensor` of shape `(N, max_seq_len)`):
                Text tokens to use for inference.
            prefix_embeds (`paddle.Tensor` of shape `(N, prefix_length, 768)`):
                Prefix embedding to prepend to the embedded tokens.
            attention_mask (`paddle.Tensor` of shape `(N, prefix_length + max_seq_len, 768)`, *optional*):
                Attention mask for the prefix embedding.
            labels (`paddle.Tensor`, *optional*):
                Labels to use for language modeling.
        """
        embedding_text = self.transformer.transformer.wte(input_ids)
        hidden = self.encode_prefix(prefix_embeds)
        prefix_embeds = self.decode_prefix(hidden)
        embedding_cat = paddle.concat((prefix_embeds, embedding_text), axis=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0])
            labels = paddle.concat(x=(dummy_token, input_ids), axis=1)
        out = self.transformer(inputs_embeds=embedding_cat, labels=labels, attention_mask=attention_mask)
        if self.prefix_hidden_dim is not None:
            return out, hidden
        else:
            return out

    def get_dummy_token(
        self,
        batch_size: int,
    ) -> paddle.Tensor:
        return paddle.zeros(shape=[batch_size, self.prefix_length], dtype="int64")

    def encode(self, prefix):
        return self.encode_prefix(prefix)

    @paddle.no_grad()
    def generate_captions(
        self,
        features,
        eos_token_id,
    ):
        """
        Generate captions given text embedding features. Returns list[L].

        Args:
            features (`paddle.Tensor` of shape `(B, L, D)`):
                Text embedding features to generate captions from.
            eos_token_id (`int`):
                The token ID of the EOS token for the text decoder model.

        Returns:
            `List[str]`: A list of strings generated from the decoder model.
        """
        # TODO junnyu, support float16
        features = features.cast(self.dtype)
        # the low dimension representation of clip feature
        features = paddle.split(features, 1, axis=0)
        generated_tokens = []
        generated_seq_lengths = []
        for feature in features:
            feature = self.decode_prefix(feature)  # back to the clip feature
            # Only support beam search for now
            output_tokens, seq_lengths = self.generate_beam(input_embeds=feature, eos_token_id=eos_token_id)
            generated_tokens.append(output_tokens[0])
            generated_seq_lengths.append(seq_lengths[0])
        generated_tokens = paddle.stack(generated_tokens)
        generated_seq_lengths = paddle.stack(generated_seq_lengths)
        return generated_tokens, generated_seq_lengths

    @paddle.no_grad()
    def generate_beam(
        self,
        input_ids=None,
        input_embeds=None,
        beam_size: int = 5,
        entry_length: int = 67,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ):
        """
        Generates text using the given tokenizer and text prompt or token embedding via beam search. This
        implementation is based on the beam search implementation from the [original UniDiffuser
        code](https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py#L89).

        Args:
            eos_token_id (`int`, *optional*):
                The token ID of the EOS token for the text decoder model.
            input_ids (`paddle.Tensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Tokenizer indices of input sequence tokens in the vocabulary. One of `input_ids` and `input_embeds`
                must be supplied.
            input_embeds (`paddle.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                An embedded representation to directly pass to the transformer as a prefix for beam search. One of
                `input_ids` and `input_embeds` must be supplied.
            beam_size (`int`, *optional*, defaults to `5`):
                The number of best states to store during beam search.
            entry_length (`int`, *optional*, defaults to `67`):
                The number of iterations to run beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The temperature to use when performing the softmax over logits from the decoding model.

        Returns:
            `Tuple(paddle.Tensor, paddle.Tensor)`: A tuple of tensors where the first element is a tensor of generated
            token sequences sorted by score in descending order, and the second element is the sequence lengths
            corresponding to those sequences.
        """
        # Generates text until stop_token is reached using beam search with the desired beam size.
        stop_token_index = eos_token_id
        tokens = None
        scores = None
        seq_lengths = paddle.ones(shape=beam_size, dtype="int32")
        is_stopped = paddle.zeros(shape=beam_size, dtype="bool")

        if input_embeds is not None:
            generated = input_embeds
        else:
            generated = self.transformer.transformer.wte(input_ids)

        for i in range(entry_length):
            outputs = self.transformer(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = paddle.nn.functional.softmax(logits, axis=-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(k=beam_size, axis=-1)
                generated = generated.expand(shape=[beam_size, *generated.shape[1:]])
                next_tokens, scores = next_tokens.transpose(perm=[1, 0]), scores.squeeze(axis=0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(shape=[beam_size, *tokens.shape[1:]])
                    tokens = paddle.concat(x=(tokens, next_tokens), axis=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                is_stopped_tensor_int32 = paddle.cast(~is_stopped, dtype='int32')
                seq_lengths += is_stopped_tensor_int32
                scores_sum_average = scores_sum / seq_lengths[:, None].cast(scores_sum.dtype)
                scores_sum_average, next_tokens = scores_sum_average.reshape([-1]).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = paddle.concat(x=(tokens, next_tokens), axis=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths.cast(scores_sum_average.dtype)
                is_stopped = paddle.cast(is_stopped, "int32")  # TODO: nf
                is_stopped = is_stopped[next_tokens_source]
                is_stopped = paddle.cast(is_stopped, "bool")

            next_token_embed = self.transformer.get_input_embeddings()(next_tokens.squeeze()).reshape(
                [generated.shape[0], 1, -1]
            )
            generated = paddle.concat((generated, next_token_embed), axis=1)
            is_stopped = paddle.bitwise_or(is_stopped, next_tokens.equal(stop_token_index).squeeze())
            if is_stopped.astype("bool").all():
                break

        scores = scores / seq_lengths.cast(scores.dtype)
        order = scores.argsort(descending=True)
        # tokens tensors are already padded to max_seq_length
        output_texts = [tokens[i] for i in order]
        output_texts = paddle.stack(output_texts, axis=0)
        seq_lengths = paddle.to_tensor(data=[seq_lengths[i] for i in order], dtype=seq_lengths.dtype)
        return output_texts, seq_lengths
