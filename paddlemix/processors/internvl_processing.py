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
"""
Processor class for InternVL.
"""
import dataclasses
import io
import json
import sys
from copy import deepcopy
from enum import IntEnum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.vision.transforms as transforms
import requests
from paddlenlp.transformers.tokenizer_utils_base import TensorType
from PIL import Image

from .base_processing import ProcessorMixin
from .processing_utils import BaseImageProcessor, BaseTextProcessor

__all__ = ["InternVL2Processor", "InternVL2ImageProcessor", "InternVL2TextProcessor"]

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
IGNORE_TOKEN_ID = "<s>"


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


class InternVL2Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "InternLM2Tokenizer"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)
        self.model_config_path = kwargs.get("model_name_or_path", None)
        self.model_config = json.loads(Path(self.model_config_path + "/config.json").read_text())

        self.tempalte_name = self.model_config.get("template", None)
        self.img_size = self.model_config.get("force_image_size", 448)
        self.max_length = self.model_config.get("max_length", 4096)

        self.pad2square = self.model_config.get("pad2square", False)
        self.group_by_length = self.model_config.get("group_by_length", False)
        self.dynamic_image_size = self.model_config.get("dynamic_image_size", False)
        self.use_thumbnail = self.model_config.get("use_thumbnail", False)

        self.min_dynamic_patch = self.model_config.get("min_dynamic_patch", 1)
        self.max_dynamic_patch = self.model_config.get("max_dynamic_patch", 6)

        self.min_num_frame = self.model_config.get("min_num_frame", 4)
        self.max_num_frame = self.model_config.get("max_num_frame", 12)

        self.sampling_method = self.model_config.get("sampling_method", "rand")
        self.normalize_type = self.model_config.get("normalize_type", "imagenet")
        self.random_seed = self.model_config.get("random_seed", 0)
        self.down_sample_ratio = kwargs.get("down_sample_ratio", 0.5)

        self.vision_config = self.model_config.get("vision_config", None)
        self.patch_size = self.vision_config.get("patch_size", 14)

        self.image_processor = InternVL2ImageProcessor(self.img_size)
        self.text_processor = InternVL2TextProcessor()

        self.num_image_token = int((self.img_size // self.patch_size) ** 2 * (self.down_sample_ratio**2))
        self.rng = np.random.default_rng(seed=self.random_seed)

    def __call__(
        self,
        query: List[dict] = None,
        record: List[dict] = None,
        mode: str = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):

        if query is None and record is None:
            raise ValueError("You have to specify query or record.")
        if query is None:
            query = record

        if mode == "train":
            inputs = self.train_preprocess(query)

        else:  # TODO: what's this for? chat?
            images = []
            for ele in query:
                if "image" in ele:
                    images.append(ele["image"])

            inputs = self.tokenizer(query, return_tensors=return_tensors)
            inputs["images"] = None

            if len(images) > 0:
                inputs["images"] = self.image_processor(images)

        return inputs

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        # you need change the name later
        if self.template_name == "Hermes-2":
            self.text_processor = self.text_processor.preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = self.text_processor.preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = self.text_processor.preprocess_phi3
        else:
            raise ValueError(f"Template name {self.template_name} is not supported.")
        return preprocess_function

    # no image here
    def pure_text_get_item(self, sources):
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        pixel_values = self.image_processor.pure_text_image_process(image)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        ret = preprocess_function(
            self.template_name,
            [deepcopy(sources["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
        )

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor([1] * num_patches, dtype=paddle.int64),
        )
        return ret

    def multi_modal_get_item(self, data_item):
        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<image>\n" + data_item["conversations"][0]["value"]

        # Merge the image path
        image_path = self.get_image_path(data_item["image"])

        pixel_values = self.image_processor.multi_modal_image_process(
            image_path, self.min_dynamic_patch, self.max_dynamic_patch, self.dynamic_image_size
        )

        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor([1] * num_patches, dtype=paddle.int64),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):

        num_image = len(data_item["image"])
        pixel_values, num_tiles = self.image_processor.multi_modal_image_process(
            num_image, self.min_dynamic_patch, self.max_dynamic_patch, self.dynamic_image_size
        )

        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            num_image=num_image,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor([1] * num_patches, dtype=paddle.int64),
        )
        return ret

    def train_preprocess(self, sources):

        # cal the token length
        conversations = "\n".join([temp["value"] for temp in sources["conversations"]])
        str_length = len(conversations)
        if str_length not in self.conv2length:
            token_length = self.tokenizer(conversations, return_tensors="pd").input_ids.size(1)
            self.conv2length[str_length] = token_length + self.num_image_token * (
                self.max_dynamic_patch + self.use_thumbnail
            )
        else:
            token_length = self.conv2length[str_length]

        if "image" in sources and len(sources["image"]) != 0:
            if type(sources["image"]) == list:
                ret = self.multi_modal_multi_image_get_item(sources)
            else:
                ret = self.multi_modal_get_item(sources)
        elif "video" in sources and sources["video"] is not None and sources["video"] != "":
            ret = self.video_get_item(sources)
        else:
            ret = self.pure_text_get_item(sources)
        return ret

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class InternVL2ImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        image_size: int = 448,
        normalize_type="imagenet",
        pad2square: bool = False,
        use_thumbnail: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size, image_size
        self.pad2square = pad2square
        self.use_thumbnail = use_thumbnail
        if normalize_type == "imagenet":
            self.image_mean = [0.485, 0.456, 0.406]
            self.image_std = [0.229, 0.224, 0.225]
        elif normalize_type == "clip":
            self.image_mean = [0.4814546, 0.4578275, 0.40821073]
            self.image_std = [0.2686295, 0.2613025, 0.2757711]
        elif normalize_type == "siglip":
            self.image_mean = [0.5, 0.5, 0.5]
            self.image_std = [0.5, 0.5, 0.5]

        self.qualities = list(range(75, 101))
        self.jpeg_degrade_functions = {quality: self.simulate_jpeg_degradation(quality) for quality in self.qualities}
        self.normalize = transforms.Normalize(self.image_mean, self.image_std)

    def simulate_jpeg_degradation(self, quality):
        def jpeg_degrade(img):
            with io.BytesIO() as output:
                img.convert("RGB").save(output, format="JPEG", quality=quality)
                output.seek(0)  # Move the reading cursor to the start of the stream
                img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
            return img_jpeg

        return jpeg_degrade

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def build_image_transform(self, is_train):
        if is_train:
            image_transform = transforms.Compose(
                [
                    transforms.RandomChoice(
                        [transforms.Lambda(self.jpeg_degrade_functions[quality]) for quality in self.qualities]
                    ),
                    transforms.Resize((self.image_size, self.image_size), interpolation="bicubic"),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            image_transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size), interpolation="bicubic"),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        return image_transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if self.use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def pure_text_image_process(self, image, min_dynamic_patch, max_dynamic_patch, is_train: bool = True):
        image = image.convert("RGB")
        if not self.is_train and self.pad2square:
            image = self.expand2square(image, tuple(int(x * 255) for x in self.image_mean))

        image_transform = self.build_image_transform(is_train=is_train)
        images = self.dynamic_preprocess(image, min_dynamic_patch, 1, image_size=self.image_size)

        pixel_values = [image_transform(image) for image in images]
        pixel_values = paddle.stack(pixel_values, axis=0)

        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        return pixel_values

    def multi_modal_image_process(
        self, image_path, min_dynamic_patch, max_dynamic_patch, dynamic_image_size, is_train: bool = True
    ):
        image_transform = self.build_image_transform(is_train=is_train)
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        image = image.convert("RGB")
        if not self.is_train and self.pad2square:
            image = self.expand2square(image, tuple(int(x * 255) for x in self.image_mean))

        if dynamic_image_size:
            images = self.dynamic_preprocess(image, min_dynamic_patch, max_dynamic_patch, image_size=self.image_size)

        else:
            images = [image]

        pixel_values = [image_transform(image) for image in images]
        pixel_values = paddle.stack(pixel_values, axis=0)
        num_patches = pixel_values.size(0)
        # Ensure there is only one patch
        if not dynamic_image_size:
            assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        return pixel_values

    def multi_modal_multi_image_process(
        self, image_paths, min_dynamic_patch, max_dynamic_patch, dynamic_image_size, is_train: bool = True
    ):
        image_transform = self.build_image_transform(is_train=is_train)
        images, num_tiles = [], []
        num_image = len(image_paths)
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            if not self.is_train and self.pad2square:
                image = self.expand2square(image, tuple(int(x * 255) for x in self.image_mean))

            if dynamic_image_size:
                image = self.dynamic_preprocess(
                    image, min_dynamic_patch, max_dynamic_patch // num_image, image_size=self.image_size
                )
                images += image
            else:
                images.append(image)
                num_tiles.append(1)
        pixel_values = [image_transform(image) for image in images]
        pixel_values = paddle.stack(pixel_values, axis=0)
        return pixel_values, num_tiles

    def __call__(self, image_paths: List[str], is_train: bool = False):
        image_transform = self.build_image_transform(is_train=is_train)
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            if not self.is_train and self.pad2square:
                image = self.expand2square(image, tuple(int(x * 255) for x in self.image_mean))
            images.append(image_transform(image))

        images = paddle.stack(x=images, axis=0)
        return images


class InternVL2TextProcessor(BaseTextProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_templates: Dict[str, Conversation] = {}
        # InternVL-Chat-V1-1 template
        self.register_conv_template(
            Conversation(
                name="internvl_zh",
                system_template="",
                roles=("<human>", "<bot>"),
                sep_style=SeparatorStyle.INTERNVL_ZH,
                sep="</s>",
                sep2=" ",
            )
        )

        # Both Hermes-2 and internlm2-chat are chatml-format conversation templates. The difference
        # is that during training, the preprocessing function for the Hermes-2 template doesn't add
        # <s> at the beginning of the tokenized sequence, while the internlm2-chat template does.
        # Therefore, they are completely equivalent during inference.
        self.register_conv_template(
            Conversation(
                name="Hermes-2",
                system_template="<|im_start|>system\n{system_message}",
                # note: The new system prompt was not used here to avoid changes in benchmark performance.
                # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。',
                system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
                roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
                sep_style=SeparatorStyle.MPT,
                sep="<|im_end|>",
                stop_token_ids=[
                    2,
                    6,
                    7,
                    8,
                ],
                stop_str="<|endoftext|>",
            )
        )

        # internlm2-chat
        self.register_conv_template(
            Conversation(
                name="internlm2-chat",
                system_template="<|im_start|>system\n{system_message}",
                # note: The new system prompt was not used here to avoid changes in benchmark performance.
                # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。',
                system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
                roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
                sep_style=SeparatorStyle.MPT,
                sep="<|im_end|>",
                stop_token_ids=[2, 92543, 92542],
            )
        )

        # phi-3
        self.register_conv_template(
            Conversation(
                name="phi3-chat",
                system_template="<|system|>\n{system_message}",
                # note: The new system prompt was not used here to avoid changes in benchmark performance.
                # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。',
                system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
                roles=("<|user|>\n", "<|assistant|>\n"),
                sep_style=SeparatorStyle.MPT,
                sep="<|end|>",
                stop_token_ids=[2, 32000, 32007],
            )
        )

    def register_conv_template(self, template: Conversation, override: bool = False):
        """Register a new conversation template."""
        if not override:
            assert template.name not in self.conv_templates, f"{template.name} has been registered."

        self.conv_templates[template.name] = template

    def get_conv_template(self, name: str) -> Conversation:
        """Get a conversation template."""
        return self.conv_templates[name].copy()

    def preprocess_internlm(
        self,
        template_name,
        sources,
        tokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
    ):
        conv = self.get_conv_template(template_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                sentence["value"] = sentence["value"].strip()
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        if not text_only:
            new_conversations = []
            for conversation in conversations:
                for i in range(num_image):
                    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                    conversation = conversation.replace("<image>", image_tokens, 1)
                new_conversations.append(conversation)
            conversations = new_conversations

        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pd",
            padding=False if group_by_length or use_packed_ds else "max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID  # <s>
            parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
            info = parts[0] + conv.roles[1]
            temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
            target[cur_len : cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len

            for index in range(1, len(parts) - 1):
                info = parts[index]
                part1, part2 = info.split(conv.roles[0])
                temp_len = len(tokenizer(part1).input_ids) - 1
                cur_len = cur_len + temp_len
                part = conv.roles[0] + part2 + conv.roles[1]
                temp_len = len(tokenizer(part).input_ids) - 1
                target[cur_len : cur_len + temp_len] = IGNORE_TOKEN_ID
                cur_len = cur_len + temp_len
            last_info = parts[-1]
            temp_len = len(tokenizer(last_info).input_ids) - 1
            cur_len = cur_len + temp_len

            target[cur_len:] = IGNORE_TOKEN_ID
            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = paddle.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                print(repr(tokenizer.decode(z)))

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.")
                    sys.stdout.flush()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def preprocess_mpt(
        self,
        template_name,
        sources,
        tokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
    ) -> Dict:
        conv = self.get_conv_template(template_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        if not text_only:
            new_conversations = []
            for conversation in conversations:
                for i in range(num_image):
                    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                    conversation = conversation.replace("<image>", image_tokens, 1)
                new_conversations.append(conversation)
            conversations = new_conversations

        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pd",
            padding=False if group_by_length or use_packed_ds else "max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep)
            re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
            for conv_idx in range(3, len(turns), 2):
                re_turns.append(conv.sep.join(turns[conv_idx : conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(re_turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids) + 1

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids)

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
                # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
                # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}."
                    )
                    sys.stdout.flush()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def preprocess_phi3(
        self,
        template_name,
        sources,
        tokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
    ) -> Dict:
        conv = self.get_conv_template(template_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        if not text_only:
            new_conversations = []
            for conversation in conversations:
                for i in range(num_image):
                    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                    conversation = conversation.replace("<image>", image_tokens, 1)
                new_conversations.append(conversation)
            conversations = new_conversations

        # Tokenize conversations
        tokenizer.padding_side = "right"
        input_ids = tokenizer(
            conversations,
            return_tensors="pd",
            padding=False if group_by_length or use_packed_ds else "max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())

            turns = conversation.split(conv.sep)
            re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
            for conv_idx in range(3, len(turns), 2):
                re_turns.append(conv.sep.join(turns[conv_idx : conv_idx + 2]))  # user + gpt
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            target[target == endoftext_id] = IGNORE_TOKEN_ID

            for i, turn in enumerate(re_turns):
                if turn == "":
                    break
                if i == 0:
                    turn_len = len(tokenizer(turn).input_ids)
                else:
                    turn_len = len(tokenizer(turn).input_ids) - 1
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if i == 0:
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                else:
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
                # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
                # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = paddle.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                print(repr(tokenizer.decode(z)))

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}."
                    )
                    sys.stdout.flush()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
