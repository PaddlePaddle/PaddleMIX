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

from typing import TYPE_CHECKING, Dict

import gradio as gr

from ..common import get_model_info, list_checkpoints
from ..extras.constants import DEFAULT, DEFAULT_TEMPLATE, METHODS, SUPPORTED_MODELS
from ..extras.template import TEMPLATES

if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> Dict[str, "Component"]:
    # available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]
    available_models = list(SUPPORTED_MODELS.keys())
    with gr.Row():
        lang = gr.Dropdown(choices=["en", "zh"], scale=1, value="en")
        model_name = gr.Dropdown(choices=available_models, scale=3, value=DEFAULT["model"])
        model_path = gr.Textbox(scale=3)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        checkpoint_path = gr.Dropdown(scale=6, value="")

    with gr.Row():
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value=DEFAULT_TEMPLATE["default"], scale=2)
    model_name.change(get_model_info, [model_name], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)

    finetuning_type.change(inputs=[finetuning_type], outputs=[finetuning_type]).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        template=template,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
    )
