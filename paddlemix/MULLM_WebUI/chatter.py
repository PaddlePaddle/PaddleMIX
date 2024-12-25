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
from threading import Thread
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple

import gradio as gr
import paddle
from paddlenlp.generation import TextIteratorStreamer
from paddlenlp.peft import LoRAModel
from paddlenlp.utils.import_utils import import_module

from ..models.qwen2_vl import MIXQwen2Tokenizer
from ..processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)
from .common import ChatState, change_checkbox, chat_ready, get_save_dir
from .extras.constants import FINAL_CHECKPOINT_NAME, MODEL_MAPPING
from .locales import ALERTS, LOCALES

if TYPE_CHECKING:
    from .manager import Manager


class WebChatModel:
    def __init__(self, manager: "Manager", demo_mode: bool = False, lazy_init: bool = True) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.engine = None
        self.processor = None
        self.tokenizer = None
        self.terminators = ["<|im_end|>"]
        # self.min_pixels = 256 * 28 * 28  # 200704
        # self.max_pixels = 1280 * 28 * 28  # 1003520

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load_model(self, data) -> Generator[str, None, None]:
        engine_cls = self.get_model(data)
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        finetuning_type = get("top.finetuning_type")
        infer_dtype = get("infer.infer_dtype")
        checkpoint_path = get("top.checkpoint_path")
        state_checkbox_group = get("infer.state_checkbox_group")
        selected_ckpt = get("infer.ckpt_box")
        if selected_ckpt == FINAL_CHECKPOINT_NAME:
            ckpt_path = os.path.join(get_save_dir(model_name, finetuning_type), checkpoint_path)
        elif selected_ckpt != "":
            ckpt_path = os.path.join(get_save_dir(model_name, finetuning_type), checkpoint_path, selected_ckpt)
        else:
            ckpt_path = ""

        error = ""
        yield ALERTS["info_loading"][lang], state_checkbox_group

        if self.loaded:
            error = ALERTS["err_exists"][lang]
            yield error, change_checkbox(state_checkbox_group, True, choice_type=LOCALES["model_tag"][lang])
            return
        elif not model_name:
            error = ALERTS["err_no_model"][lang]
        elif not model_path:
            error = ALERTS["err_no_path"][lang]

        self.engine = engine_cls.from_pretrained(model_path, dtype=infer_dtype)
        self.processor = self.get_processor(model_path)

        # load lora
        if ckpt_path != "":
            self.engine = LoRAModel.from_pretrained(model=self.engine, lora_path=ckpt_path)

        if error:
            gr.Warning(error)
            yield error
            return

        yield ALERTS["info_loaded"][lang], change_checkbox(
            state_checkbox_group, True, choice_type=LOCALES["model_tag"][lang]
        )

    def unload_model(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]
        state_checkbox_group = data[self.manager.get_elem_by_id("infer.state_checkbox_group")]

        if not self.loaded:
            yield ALERTS["info_unload_error"][lang], state_checkbox_group
            return

        yield ALERTS["info_unloading"][lang], state_checkbox_group
        self.engine = None
        state_checkbox_group.remove(LOCALES["model_tag"][lang])
        paddle.device.cuda.empty_cache()
        yield ALERTS["info_unloaded"][lang], state_checkbox_group

    def multi_round_chat(
        self,
        lang,
        chatbot,
        messages,
        question_box,
        question_type,
        image,
        video,
        chat_checkbox,
        max_new_tokens,
        top_p,
        temperature,
        seed,
        info_box,
    ) -> Tuple[List[List[Optional[str]]], List[Dict[str, str]], str]:
        chat_state = {
            "model": LOCALES["model_tag"][lang],
            "image": LOCALES["image_tag"][lang],
            "video": LOCALES["video_tag"][lang],
            "question": LOCALES["question_tag"][lang],
        }

        check_result = chat_ready(chat_checkbox, chat_state)
        if check_result == ChatState.MISSING_QUESTION:
            yield chatbot, messages, gr.update(value=question_box), gr.update(
                value=ALERTS["info_query"][lang]
            ), gr.update(interactive=True)
            return
        if check_result == ChatState.MISSING_MODEL:
            yield chatbot, messages, gr.update(value=question_box), gr.update(
                value=ALERTS["info_upload_model"][lang]
            ), gr.update(interactive=True)
            return
        if check_result == ChatState.MISSING_FILE:
            yield chatbot, messages, gr.update(value=question_box), gr.update(
                value=ALERTS["info_upload_file"][lang]
            ), gr.update(interactive=True)
            return
        msg = {
            "role": "user",
            "content": [],
        }
        last_img_inp = None
        last_video_inp = None

        # find last image and video input
        for m in messages[::-1]:
            for content in m["content"]:
                if "video" in content.keys():
                    last_video_inp = content["video"]

                if "image" in content.keys():
                    last_img_inp = content["image"]
            if last_img_inp is not None and last_video_inp is not None:
                break

        if image is not None and image == last_img_inp:
            image = None

        if video is not None and video == last_video_inp:
            video = None

        if question_type == "image" and image is not None:
            msg["content"].append({"type": "image", "image": image})

        if question_type == "video" and video is not None:
            msg["content"].append({"type": "video", "video": video, "fps": 1, "max_pixels": 360 * 420})

        chatbot += [[question_box, None]]
        msg["content"].append({"type": "text", "text": f"{question_box}"})

        messages.append(msg)
        paddle.seed(seed=seed)
        generate_cfg = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        response = ""
        res = self.generate(messages, generate_cfg)
        for text in res:
            response += text
            yield chatbot + [[None, response]], messages + [
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ], gr.update(value=question_box), gr.update(value=ALERTS["info_generating"][lang]), gr.update(
                interactive=False
            )

        yield chatbot + [[None, response]], messages + [
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ], gr.update(value=question_box), gr.update(value=ALERTS["info_generated"][lang]), gr.update(interactive=True)

    def get_model(self, data):
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name = get("top.model_name")
        model_module = import_module(f"paddlemix.models.{MODEL_MAPPING[model_name]}")

        return model_module

    def get_processor(self, model_path):
        image_processor = Qwen2VLImageProcessor()
        tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
        # processor = Qwen2VLProcessor(image_processor, tokenizer,min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        processor = Qwen2VLProcessor(image_processor, tokenizer)
        self.tokenizer = tokenizer

        return processor

    def generate(self, messages, generate_cfg):
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )

        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_special_tokens=True)
        generation_kwargs = {
            "streamer": streamer,
        }

        generation_kwargs.update(generate_cfg)
        generation_kwargs.update(inputs)

        thread = Thread(target=self.engine.generate, kwargs=generation_kwargs)
        """Class Method: *.start, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        thread.start()
        return streamer
