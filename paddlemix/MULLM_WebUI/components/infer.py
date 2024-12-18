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

from ..common import change_checkbox, list_checkpoint_item
from ..extras.packages import is_gradio_available
from .chatbot import enable_checkpoint_box

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_infer_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    checkpoint_path: "gr.Dropdown" = engine.manager.get_elem_by_id("top.checkpoint_path")

    elem_dict = dict()

    with gr.Row():
        infer_dtype = gr.Dropdown(choices=["float16", "bfloat16", "float32"], value="float16")
        ckpt_box = gr.Dropdown(value="", visible=False)

    with gr.Row():
        load_btn = gr.Button()
        unload_btn = gr.Button()

    with gr.Row():
        with gr.Column():
            with gr.Tab("Image"):
                image = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"])

            with gr.Tab("Video"):
                video = gr.Video(sources=["upload"])

            state_checkbox_group = gr.CheckboxGroup(value=[], interactive=False)
            info_box = gr.Textbox(show_label=True, interactive=False)

        with gr.Column(scale=1):
            question_box = gr.Textbox(value="", interactive=True)
            question_type = gr.Dropdown(choices=["image", "video"], value="image")
            seed_box = gr.Textbox(value=42, interactive=True)
            max_new_tokens = gr.Slider(minimum=8, maximum=4096, value=512, step=1)
            top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.7, step=0.01)
            temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01)
            chat_btn = gr.Button()
            clear_btn = gr.Button()

    chatbot = gr.Chatbot(show_copy_button=True)
    messages = gr.State([])

    input_elems.update({image})
    input_elems.update({video})
    input_elems.update({chatbot})
    input_elems.update({question_box, info_box})
    input_elems.update({messages})
    input_elems.update({infer_dtype, ckpt_box})
    input_elems.update({state_checkbox_group})
    input_elems.update({question_type, max_new_tokens, top_p, temperature, seed_box})
    elem_dict.update(
        dict(
            infer_dtype=infer_dtype,
            ckpt_box=ckpt_box,
            load_btn=load_btn,
            unload_btn=unload_btn,
            info_box=info_box,
            question_box=question_box,
            seed_box=seed_box,
            question_type=question_type,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            chat_btn=chat_btn,
            clear_btn=clear_btn,
            state_checkbox_group=state_checkbox_group,
            image=image,
            video=video,
            chatbot=chatbot,
            messages=messages,
        )
    )

    # Button
    load_btn.click(engine.chatter.load_model, input_elems, [info_box, state_checkbox_group])
    unload_btn.click(engine.chatter.unload_model, input_elems, [info_box, state_checkbox_group])

    clear_btn.click(lambda: ([], [], "", None, None), outputs=[chatbot, messages, question_box, image, video])

    chat_btn.click(
        engine.chatter.multi_round_chat,
        inputs=[
            engine.manager._id_to_elem["top.lang"],
            chatbot,
            messages,
            question_box,
            question_type,
            image,
            video,
            state_checkbox_group,
            max_new_tokens,
            top_p,
            temperature,
            seed_box,
            info_box,
        ],
        outputs=[chatbot, messages, question_box, info_box, chat_btn],
    )
    question_box.change(
        change_checkbox,
        inputs=[state_checkbox_group, question_box, engine.manager._id_to_elem["top.lang"], gr.State("question_tag")],
        outputs=state_checkbox_group,
        every=3,
    )

    image.change(
        change_checkbox,
        inputs=[state_checkbox_group, image, engine.manager._id_to_elem["top.lang"], gr.State("image_tag")],
        outputs=state_checkbox_group,
    )
    video.change(
        change_checkbox,
        inputs=[state_checkbox_group, video, engine.manager._id_to_elem["top.lang"], gr.State("video_tag")],
        outputs=state_checkbox_group,
    )
    checkpoint_path.change(
        list_checkpoint_item,
        [
            engine.manager._id_to_elem["top.model_name"],
            engine.manager._id_to_elem["top.finetuning_type"],
            engine.manager._id_to_elem["top.checkpoint_path"],
        ],
        [ckpt_box],
        queue=False,
    ).then(enable_checkpoint_box, inputs=[checkpoint_path], outputs=[ckpt_box], show_progress=False)

    return elem_dict
