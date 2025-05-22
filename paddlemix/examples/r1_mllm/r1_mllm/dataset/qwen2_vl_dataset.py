# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import paddle
import yaml
from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.transformers.processing_utils import ProcessorMixin
from paddlenlp.utils.import_utils import import_module
from PIL import Image

from paddlemix.models.qwen2_vl.template import TEMPLATES
from paddlemix.processors.qwen2_5_vl_processing import process_vision_info


@dataclass
class Qwen2VLDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    template: Optional["TEMPLATES"] = None
    processor: Optional["ProcessorMixin"] = None
    process_vision_info = None
    template_name: Optional[str] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")
        if self.process_vision_info is None:
            self.process_vision_info = import_module(f"paddlemix.processors.{self.template_name}_processing")

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "paddle.Tensor"]:
        batched_pixel_values = []
        batched_attention_mask = []
        batched_input_ids = []
        batched_labels = []
        batched_image_grid_thw = []
        for feature in features:
            messages = feature["prompt"]
            solution_text = feature["solution"]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            messages[0]["content"][0]["image"] = feature["image"]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=prompt_text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pd",
            )

            solution_inputs = self.tokenizer(
                text=solution_text,
                return_tensors="pd",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            batched_pixel_values.append(inputs["pixel_values"])
            batched_input_ids.append(inputs["input_ids"][0])
            batched_attention_mask.append(inputs["attention_mask"][0])
            batched_labels.append(solution_inputs["input_ids"][0])
            batched_image_grid_thw.append(inputs["image_grid_thw"])
        return {
            "pixel_values": paddle.stack(batched_pixel_values),
            "attention_mask": paddle.stack(batched_attention_mask),
            "input_ids": paddle.stack(batched_input_ids),
            "labels": paddle.stack(batched_labels),
            "image_grid_thw": paddle.stack(batched_image_grid_thw),
        }


class Qwen2VLRECDataset(paddle.io.Dataset):
    def __init__(self, data_path: str, script_args, training_args, model_args, tokenizer, processor, template):
        super(Qwen2VLRECDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.tokenizer = tokenizer
        self.processor = processor
        self.template = template
        self.training_args = training_args
        self.model_args = model_args
        self.script_args = script_args

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")
                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def _preprocess_image(self, image, image_max_pixels, image_min_pixels):
        r"""
        Pre-processes a single image.
        """
        image = self.template.mm_plugin._preprocess_image(
            image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
        )
        return image

    def get_image_path(self, image_path):
        return image_path

    def get_transform(self):
        return self.processor.image_processor

    def multi_modal_get_item(self, data_item):
        messages = data_item["messages"]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        label_ids = self.processor.tokenizer(
            text=str(data_item["label"]),
            padding=True,
            padding_side="left",
            return_tensors="pd",
        )
        # unwrap
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]

        # Create the final return dictionary
        ret = dict(
            **inputs,
            labels=label_ids["input_ids"][0],
        )
        return ret

    def __getitem__(self, i):
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

        def make_conversation_image(example, image):
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": QUESTION_TEMPLATE.format(Question=example["problem"]),
                            },
                        ],
                    }
                ]
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example["image"])
            image = self._preprocess_image(
                Image.open(image_path).convert("RGB"),
                image_max_pixels=self.script_args.max_pixels,
                image_min_pixels=self.script_args.min_pixels,
            )
        else:
            image = None
        data_item = {
            "image": image,
            "image_path": example["image"],
            "label": example["solution"],
            "messages": make_conversation_image(example, image)["messages"],
        }
        return self.multi_modal_get_item(data_item)
