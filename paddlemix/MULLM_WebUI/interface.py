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

import os

from .components import create_infer_tab, create_top, create_train_tab
from .css import CSS
from .engine import Engine
from .extras.packages import is_gradio_available

if is_gradio_available():
    import gradio as gr


def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)

    with gr.Blocks(title="PaddleMIX Board", css=CSS) as demo:
        engine.manager.add_elems("top", create_top())
        lang: "gr.Dropdown" = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("🔥 Train"):
            engine.manager.add_elems("train", create_train_tab(engine))

        with gr.Tab("🔑 Chat"):
            engine.manager.add_elems("infer", create_infer_tab(engine))

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)

        # lang.input(inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = os.getenv("GRADIO_IPV6", "0").lower() in ["true", "1"]
    gradio_share = os.getenv("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "8260"))
    demo = create_ui()
    demo.queue()
    demo.launch(share=gradio_share, server_name=server_name, inbrowser=True, server_port=server_port, max_threads=5)
