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
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import List, Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.nn import CrossEntropyLoss
from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import LlamaForCausalLM, Qwen2ForCausalLM
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast

# from paddlenlp.transformers.model_utils import PretrainedModel
from paddlemix.models.model_utils import MixPretrainedModel, NPUCrossEntropyLoss
from paddlemix.utils.log import logger

from ..conversation import get_conv_template
from ..internlm2.modeling_internlm2 import InternLM2ForCausalLM
from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

# from ..phi3.modeling_phi3 import Phi3ForCausalLM


__all__ = [
    "InternVLChatModel",
]


class InternVLChatModel(MixPretrainedModel):
    config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["InternVisionModel", "LlamaDecoderLayer", "InternLM2DecoderLayer", "Qwen2DecoderLayer"]
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
                self.language_model = InternLM2ForCausalLM(config.llm_config)  # [2048, 92553]
            # elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            #     self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)  # [151655, 896]
            else:
                raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
        self,
        pixel_values: paddle.Tensor,  # [14, 3, 448, 448]
        input_ids: paddle.Tensor = None,  # [2, 1918]
        attention_mask: Optional[paddle.Tensor] = None,  # [2, 1918]
        position_ids: Optional[paddle.Tensor] = None,
        image_flags: Optional[paddle.Tensor] = None,  # [14]
        past_key_values: Optional[List[paddle.Tensor]] = None,
        labels: Optional[paddle.Tensor] = None,  # [2, 1918]
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        # [2, 1918, 2048] -3972.55566406  bfloat16
        vit_embeds = self.extract_feature(
            pixel_values
        )  # pixel_values float32 [14, 3, 448, 448] sum=891674.56250000 mean=0.10577939
        # [14, 256, 2048] bfloat16 -85460.1875 -0.01165771484375

        # [14, 256, 2048] bfloat16 -87335.3984375 -0.01190185546875
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape([B * N, C])

        if paddle.distributed.get_rank() == 0:
            logger.info(
                f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )

        input_ids = input_ids.reshape([B * N])  # [3836] sum 346393658
        selected = input_ids == self.img_context_token_id  # [3836] sum 3584
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape([-1, C])
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape([-1, C])
            logger.info(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape([B, N, C])
        # [2, 1918, 2048] bfloat16 -87403.26562500

        outputs = self.language_model(
            inputs_embeds=input_embeds,  # [2, 1918, 2048]  -87403.26562500
            attention_mask=attention_mask,  # [2, 1918]  3836
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = outputs.logits
        # [2, 1918, 92553] bfloat16 -35019332.0 -0.09863674640655518

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = NPUCrossEntropyLoss() if "npu" in paddle.get_device() else CrossEntropyLoss()
            shift_logits = shift_logits.reshape([-1, self.language_model.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape([n, w, int(h * scale_factor), int(c / scale_factor)])
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.transpose([0, 2, 1, 3])
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.reshape([n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))])
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.transpose([0, 2, 1, 3])
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape([vit_embeds.shape[0], h, w, -1])
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape([vit_embeds.shape[0], -1, vit_embeds.shape[-1]])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            logger.info("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            logger.info("Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.")

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.info(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pd", padding=True)  # padding
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config["eos_token_id"] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_config
        )
        responses = tokenizer.batch_decode(generation_output[0], skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.info(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pd")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,  # [7, 3, 448, 448]
            input_ids=input_ids,  # [1, 1847]
            attention_mask=attention_mask,  # [1, 1847]
            **generation_config,  # {'max_new_tokens': 1024, 'do_sample': False, 'eos_token_id': 92542}
        )
        response = tokenizer.batch_decode(generation_output[0], skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
            if verbose:
                logger.info(query_to_print, response)
            return response

    @paddle.no_grad()
    def generate(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        visual_features: Optional[paddle.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> paddle.Tensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            # vit_embeds.shape  [7, 256, 896]  [7, 256, 2048]
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # 1b qwen2: nn.Embedding(151655, 896, sparse=False)
            # 2b internlm2: nn.Embedding(92553, 2048, padding_idx=2, sparse=False)

            B, N, C = input_embeds.shape  # [1, 5432, 896]  [1, 1847, 2048]
            input_embeds = input_embeds.reshape([B * N, C])  # [5432, 896]  [1847, 2048]

            input_ids = input_ids.reshape([B * N])  # [5432]  [1847]
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0, "None after  selected = input_ids == self.img_context_token_id"

            input_embeds[selected] = vit_embeds.reshape([-1, C])  # [7, 256, 896] -> [1792, 896]

            input_embeds = input_embeds.reshape([B, N, C])
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        ###  must add position_ids, paddlenlp bug
        if isinstance(self.language_model, Qwen2ForCausalLM):
            batch_size, seq_length = attention_mask.shape
            position_ids = paddle.arange(seq_length).expand((batch_size, seq_length))
            outputs = self.language_model.generate(
                position_ids=position_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )
        ###
        else:
            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )

        return outputs
