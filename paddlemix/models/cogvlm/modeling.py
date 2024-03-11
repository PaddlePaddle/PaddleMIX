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


import paddle

"""largely copy from llama and adapt for cogvlm"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import paddlenlp.transformers as transformers
from einops import rearrange
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration import CogVLMConfig
from .visual import EVA2CLIPModel

if TYPE_CHECKING:
    from paddlenlp.transformers.utils import ModelOutput

from ..cogagent.modeling import (
    RMSNorm,
    VisionExpertAttention,
    VisionExpertMLP,
    _expand_mask,
    _make_causal_mask,
    build_position_ids,
    is_empty,
)

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class CogVLMDecoderLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config=config)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        token_type_ids: paddle.Tensor,
        position_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class CogVLMPreTrainedModel(transformers.PretrainedModel):
    config_class = CogVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogVLMDecoderLayer", "TransformerLayer"]
    _skip_keys_device_placement = "past_key_values"


class CogVLMModel(CogVLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[CogVLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vision = EVA2CLIPModel(config)
        self.gradient_checkpointing = False

    def encode_images(self, images: List[List[paddle.Tensor]]) -> paddle.Tensor:
        images_list, images = images, []
        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)
        images = paddle.stack(x=images)
        images_features = self.vision(images)
        return images_features

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        images: List[List[paddle.Tensor]] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""
        if past_key_values is not None:
            pass
        else:
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):
                assert token_type_ids is not None, "multi-modality requires `token_type_ids`!"
                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(images)
                images_features = rearrange(images_features, "b n d -> (b n) d")
                images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.place)
                inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
            else:
                if token_type_ids is None:
                    token_type_ids = paddle.ones_like(x=input_ids, dtype="int64") * LANGUAGE_TOKEN_TYPE
                assert (
                    not (token_type_ids == VISION_TOKEN_TYPE).astype("bool").any()
                ), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None
        return self.llm_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
        self,
        input_ids: paddle.Tensor = None,
        token_type_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            position_ids = paddle.arange(
                start=past_key_values_length, end=seq_length + past_key_values_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(axis=0).reshape([-1, seq_length])
        else:
            position_ids = position_ids.reshape([-1, seq_length]).astype(dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = paddle.ones(shape=(batch_size, seq_length_with_past), dtype="bool")
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return transformers.model_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.place,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask


def _history_to_prompt(signal_type, history, query):
    if signal_type == "base":
        return query
    elif signal_type == "vqa":
        answer_format = "Short answer:"
    elif signal_type == "chat":
        answer_format = "Answer:"
    else:
        assert False, f"Unknown signal type {signal_type}"
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "Question: " + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += "Question: {} {}".format(query, answer_format)
    return prompt


class CogVLMForCausalLM(CogVLMPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.model = CogVLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        images: List[List[paddle.Tensor]] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.astype(dtype="float32")

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = paddle.nn.CrossEntropyLoss()
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape(-1)
            shift_labels = shift_labels.to(shift_logits.place)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return transformers.model_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_attention_mask_for_generation(
        self, inputs: paddle.Tensor, pad_token_id: Optional[int], eos_token_id: Optional[Union[int, List[int]]]
    ) -> paddle.Tensor:
        return paddle.ones(shape=inputs.shape[:2], dtype="int64")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        token_type_ids,
        images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = build_position_ids(token_type_ids, attention_mask)
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "images": images,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: "ModelOutput",
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = (
                paddle.ones(shape=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype) * LANGUAGE_TOKEN_TYPE
            )
            model_kwargs["token_type_ids"] = paddle.concat(x=[token_type_ids, new_token_type_ids], axis=-1)
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    x=[attention_mask, paddle.ones(shape=(attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
                    axis=-1,
                )
        elif "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = paddle.concat(
                x=[
                    decoder_attention_mask,
                    paddle.ones(shape=(decoder_attention_mask.shape[0], 1), dtype=decoder_attention_mask.dtype),
                ],
                axis=-1,
            )
        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(axis=0, index=beam_idx.to(past_state.place)) for past_state in layer_past
                ),
            )
        return reordered_past

    def build_conversation_input_ids(
        self,
        tokenizer,
        *,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        images=None,
        template_version=None
    ):
        image_size: int = self.config.vision_config["image_size"]
        patch_size: int = self.config.vision_config["patch_size"]
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, "not support multi images by now."
        history = history or []
        text = _history_to_prompt(template_version, history, query)
        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            transform = paddle.vision.transforms.Compose(
                [
                    paddle.vision.transforms.Resize((image_size, image_size), interpolation="bicubic"),
                    paddle.vision.transforms.ToTensor(),
                    paddle.vision.transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
            images = [transform(images[0])]
            vision_token_num = image_size // patch_size * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)["input_ids"]
        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": paddle.to_tensor(data=input_ids, dtype="int64"),
            "token_type_ids": paddle.to_tensor(data=token_type_ids, dtype="int64"),
            "attention_mask": paddle.to_tensor(data=attention_mask, dtype="int64"),
            "images": images,
        }
