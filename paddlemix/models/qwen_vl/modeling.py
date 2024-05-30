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

from typing import List, Optional, Tuple, Union

import paddle
from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import AutoConfig, AutoModel, PretrainedTokenizer
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel

try:
    from paddlenlp.transformers.qwen.modeling import QWenPretrainedModel
except:
    from paddlenlp.transformers.qwen.modeling import QWenPreTrainedModel as QWenPretrainedModel

from paddlemix.utils.log import logger

from ..groundingdino.utils import masked_fill
from .generation_utils import (
    HistoryType,
    StopWordsLogitsProcessor,
    decode_tokens,
    get_stop_words_ids,
    make_context,
)
from .visual import Vision

_CHECKPOINT_FOR_DOC = "qwen"
_CONFIG_FOR_DOC = "QWenConfig"
QWen_PRETRAINED_MODEL_ARCHIVE_LIST = ["qwen-7b"]
_ERROR_BAD_CHAT_FORMAT = """We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
"""
_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
"""
apply_rotary_emb_func = None
rms_norm = None


def _make_causal_mask(input_ids_shape: list, dtype: paddle.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=paddle.finfo(dtype).min)
    mask_cond = paddle.arange(end=mask.shape[-1])

    mask = masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)
    mask = mask.astype(dtype)
    if past_key_values_length > 0:
        mask = paddle.concat(x=[paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[(None), (None), :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])


def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, (None), (None), :].expand(shape=[bsz, 1, tgt_len, src_len]).astype(dtype)
    inverted_mask = 1.0 - expanded_mask

    return masked_fill(inverted_mask, inverted_mask.astype("bool"), paddle.finfo(dtype).min)


class QWen(PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        llm_config = AutoConfig.from_pretrained(config.llm_pretrained_model_name_or_path)
        self.llm = AutoModel.from_config(config=llm_config, dtype=config.dtype)
        self.recompute = config.recompute if hasattr(config, "recompute") else False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        img_pos: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.llm.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.llm.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.llm.config.use_cache
        return_dict = return_dict if return_dict is not None else self.llm.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1, input_shape[-1]])
        if position_ids is not None:
            position_ids = position_ids.reshape([-1, input_shape[-1]])
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.llm.h))
        else:
            past_length = past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = paddle.arange(start=past_length, end=input_shape[-1] + past_length, dtype="int64")
            position_ids = position_ids.unsqueeze(axis=0).reshape([-1, input_shape[-1]])

        encoder_attention_mask = None
        if inputs_embeds is None:
            inputs_embeds = self.llm.wte(input_ids)

        hidden_states = inputs_embeds

        # bool 4D mask
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_length)

        hidden_states = self.llm.drop(hidden_states)

        if images is not None:
            hidden_states_dtype = hidden_states.dtype
            if hidden_states_dtype in {paddle.bfloat16, paddle.float16}:
                hidden_states = paddle.cast(hidden_states, paddle.float32)
                images = paddle.cast(images, paddle.float32)

            for idx, (i, a, b) in enumerate(img_pos):
                index = paddle.arange(a + 1, b).unsqueeze(-1)
                hidden_states[i] = paddle.scatter(hidden_states[i], index, images[idx].astype(hidden_states.dtype))

            if hidden_states_dtype in {paddle.bfloat16, paddle.float16}:
                hidden_states = paddle.cast(hidden_states, hidden_states_dtype)

        output_shape = input_shape + [
            hidden_states.shape[-1],
        ]

        if self.recompute and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with recompute")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.llm.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.recompute and self.training:
                hidden_states.stop_gradient = False
                outputs = self.llm.recompute_training(
                    block,
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if type(outputs) is tuple:
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.llm.ln_f(hidden_states)
        hidden_states = hidden_states.reshape(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QWenLMHeadModel(QWenPretrainedModel):
    _keys_to_ignore_on_load_missing = ["h\\.\\d+\\.attn\\.rotary_emb\\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = ["h\\.\\d+\\.attn\\.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.visual = Vision(config.visual)
        self.transformer = QWen(config)
        self.lm_head = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False
        )

    def freeze_vit(self):
        for name, param in self.visual.named_parameters():
            if "attn_pool" in name:
                continue
            param.stop_gradient = True

        logger.info("freeze visual ")

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        images = kwargs.get("images", None)
        if past_key_values:
            input_ids = input_ids[:, (-1)].unsqueeze(axis=-1)
            images = None

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, (-1)].unsqueeze(axis=-1)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.astype(dtype="int64").cumsum(axis=-1) - 1
            position_ids = masked_fill(position_ids, attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, (-1)].unsqueeze(axis=-1)
        else:
            position_ids = None
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": images,
            }
        )
        return model_inputs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype(paddle.int64)
        else:
            attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        return attention_mask

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        img_pos = None

        if past_key_values is None and paddle.any(x=input_ids == self.config.visual["image_start_id"]):
            bos_pos = paddle.where(input_ids == self.config.visual["image_start_id"])
            eos_pos = paddle.where(input_ids == self.config.visual["image_start_id"] + 1)
            assert (bos_pos[0] == eos_pos[0]).astype("bool").all()
            img_pos = paddle.concat(x=(bos_pos[0], bos_pos[1], eos_pos[1]), axis=1)

            if images is None:
                images = []
                for i, a, b in img_pos:
                    image = input_ids[i][a + 1 : b - 1].tolist()
                    image = image[: image.index(self.config.visual["image_start_id"] + 2)]
                    images.append(bytes(image).decode("utf-8"))

                images = self.visual.prepare(images)

            images = self.visual(images)

            assert images.shape[0] == len(images)
        else:
            images = None

        llm_outputs = self.transformer(
            input_ids,
            images=images,
            img_pos=img_pos,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = llm_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[(...), :-1, :]
            if shift_logits.dtype == paddle.bfloat16:
                shift_logits = paddle.cast(shift_logits, paddle.float32)
            shift_labels = labels[(...), 1:]
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape([-1, shift_logits.shape[-1]]), shift_labels.reshape([-1]))

        if not return_dict:
            output = (lm_logits,) + llm_outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=llm_outputs.past_key_values,
            hidden_states=llm_outputs.hidden_states,
            attentions=llm_outputs.attentions,
        )

    def chat(
        self,
        tokenizer: PretrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Tuple[str, HistoryType]:
        generation_config = generation_config if generation_config is not None else self.generation_config
        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        assert generation_config.chat_format == "chatml", _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []
        max_window_size = kwargs.get("max_window_size", None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )
        stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format, tokenizer))

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            logits_processors = [stop_words_logits_processor]
        else:
            logits_processors = None
        input_ids = paddle.to_tensor(data=[context_tokens])

        outputs, _ = self.generate(
            input_ids, logits_processors=logits_processors, generation_config=generation_config, **kwargs
        )

        outputs = paddle.concat(x=[input_ids, outputs], axis=1)
        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors="replace",
        )
        if append_history:
            history.append((query, response))
        return response, history
