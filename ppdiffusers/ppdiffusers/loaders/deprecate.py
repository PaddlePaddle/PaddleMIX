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

from ..utils import deprecate
from ..utils.import_utils import is_paddlenlp_available


def text_encoder_lora_state_dict(text_encoder):
    deprecate(
        "text_encoder_load_state_dict in `models`",
        "0.45.0",
        "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",
    )
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


if is_paddlenlp_available():

    def text_encoder_attn_modules(text_encoder):
        deprecate(
            "text_encoder_attn_modules in `models`",
            "0.45.0",
            "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",
        )
        from ppdiffusers.transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))
        else:
            raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

        return attn_modules
