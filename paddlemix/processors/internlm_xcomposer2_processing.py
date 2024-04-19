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
Processor class for InternLM-XComposer2.
"""
from typing import List, Optional, Union

import numpy as np
import paddle
from paddle.vision.transforms import functional as F
import paddle.vision.transforms as transforms 
from paddlenlp.transformers.tokenizer_utils_base import TensorType
from PIL import Image
import requests

from .base_processing import ProcessorMixin
from .processing_utils import BaseImageProcessor, BaseTextProcessor



__all__ = [
    "InternLMXComposer2Processor",
    "InternLMXComposer2ImageProcessor",
    "InternLMXComposer2TextProcessor"
]

class InternLMXComposer2Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "InternLMXComposer2Tokenizer"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)
        img_size = kwargs.get("img_size", 224)
        self.image_processor = InternLMXComposer2ImageProcessor(img_size)
        self.text_processor = InternLMXComposer2TextProcessor()

    def __call__(
        self,
        query: List[dict] = None,
        record: List[dict] = None,
        mode: str = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        # import pdb
        # pdb.set_trace()

        if query is None and record is None:
            raise ValueError("You have to specify query or record.")
        if query is None:
            query = record

        if mode == "train":
            inputs = self.train_preprocess(query)

        else: # TODO: what's this for? chat?
            images = []
            for ele in query:
                if "image" in ele:
                    images.append(ele["image"])

            # query = self.tokenizer.from_list_format(query)
            inputs = self.tokenizer(query, return_tensors=return_tensors)
            # inputs["images"] = None
            # import pdb
            # pdb.set_trace()
            if len(images) > 0:
                inputs["images"] = self.image_processor(images)

        return inputs

    def train_preprocess(self, sources, system_message: str = "You are a helpful assistant."):
        # import pdb
        # pdb.set_trace()
        # IGNORE_TOKEN_ID = -100
        # im_start = self.tokenizer.im_start_id
        # im_end = self.tokenizer.im_end_id
        # nl_tokens = self.tokenizer("\n").input_ids
        # _system = self.tokenizer("system").input_ids + nl_tokens

        # input_id, target = [], []
        # input_id = []
        # system = [im_start] + _system + self.tokenizer(system_message).input_ids + [im_end] + nl_tokens
        # input_id += system
        # target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        # assert len(input_id) == len(target)

        import re

        image_pattern = re.compile(r"<img>.*</img>")
        image_path = []
        # import pdb
        # pdb.set_trace()
        if isinstance(sources, dict) and "conversations" in sources.keys():
            sources = sources["conversations"][0]

            sources = self.text_processor(sources)
        if "<img>" in sources:
            result = image_pattern.findall(sources)
            for ele in result:
                image_path.append(ele[5:-6])

        # input_id_conversation = self.tokenizer(sources).input_ids
        # input_id += input_id_conversation

        # im_start_index = np.where(np.array(input_id_conversation) == im_start)[0]
        # im_end_index = np.where(np.array(input_id_conversation) == im_end)[0]
        # index_list = list(zip(im_start_index, im_end_index))

        # for i in range(0, len(index_list), 2):
        #     q = index_list[i]
        #     a = index_list[i + 1]

        #     target += [im_start] + [IGNORE_TOKEN_ID] * (q[1] - q[0] + 2 - 3) + [im_end] + nl_tokens
        #     target += (
        #         [im_start]
        #         + [IGNORE_TOKEN_ID] * len(self.tokenizer("<|im_start|>assistant").input_ids)
        #         + input_id_conversation[a[0] : a[1] + 2][
        #             len(self.tokenizer("<|im_start|>assistant").input_ids) + 1 : -2
        #         ]
        #         + [im_end]
        #         + nl_tokens
        #     )

        # assert len(input_id) == len(target)

        inputs = dict(
            # input_ids=input_id[: self.max_len],
            # labels=target[: self.max_len],
            text_input=sources
        )

        text = inputs['text_input']
        if len(image_path) > 0:
            # to_regress_embeds, attention_mask, targets, im_mask = (self.interleav_wrap(image, text))
            text_tokens, text = self.interleav_wrap(text, image_path)  # just a text tokenize process
        else:  # TODO: adjust for pure text input
            text_tokens = self.tokenizer(text, return_tensors='pd',
                padding='longest', truncation=True, max_length=self.max_length,
                add_special_tokens=True)
            # to_regress_tokens, targets = self.text2emb(text, add_special=True)
            # to_regress_embeds = self.model.tok_embeddings(to_regress_tokens.input_ids)
            # attention_mask = to_regress_tokens.attention_mask
            # im_mask = paddle.zeros(shape=to_regress_embeds.shape[:2])
        # inputs_embeds = to_regress_embeds[:, :self.max_length]
        # attention_mask = attention_mask[:, :self.max_length]
        # targets = targets[:, :self.max_length]
        # im_mask = im_mask[:, :self.max_length].astype(dtype='bool')
        # labels = targets
        inputs = {
            'input_tokens': text_tokens,
            'input_text': text,
        }
        # import pdb
        # pdb.set_trace()
        if len(image_path) > 0:
            inputs["images"] = self.image_processor(image_path)

        return inputs

    # def interleav_wrap(self, img_list, text_list):
    def interleav_wrap(self, text, img_path_list):
        # wrap_embeds_list, wrap_atts_list = [], []
        # wrap_target_list, wrap_im_mask_list = [], []
        # for image, text in zip(img_list, text_list):
        # img_embeds, atts_img, img_target = self.img2emb(image)
        # text = text[0]
        # import pdb
        # pdb.set_trace()
        img_path_list = [f'<img>{p}</img>' for p in img_path_list]
        parts = text
        for img_path in img_path_list:
            parts = parts.replace(img_path, '<ImageHere>')
        text = parts
        parts = text.split('<ImageHere>')

        # wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
        wrap_tokens = []
        # temp_len = 0
        # image_nums, im_len = img_embeds.shape[:2]
        need_bos = True
        for idx, part in enumerate(parts):
            if len(part) > 0:
                # import pdb
                # pdb.set_trace()
                part_tokens = self.tokenizer(
                    part, 
                    return_tensors='pd',
                    padding='longest', 
                    add_special_tokens=need_bos)
                if need_bos:
                    need_bos = False
                wrap_tokens.append(part_tokens)
                # wrap_tokens.append(part_tokens.input_ids)
                # part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                # wrap_embeds.append(part_embeds)
                # wrap_atts.append(part_tokens.attention_mask)
                # wrap_im_mask.append(paddle.zeros(shape=part_embeds.shape[:2]).to('float32'))
                # temp_len += part_embeds.shape[1]
            # if idx < image_nums:
            #     wrap_tokens.append(img_target[idx].unsqueeze(axis=0))
            #     wrap_embeds.append(img_embeds[idx].unsqueeze(axis=0))
            #     wrap_atts.append(atts_img[idx].unsqueeze(axis=0))
            #     wrap_im_mask.append(paddle.ones_like(x=atts_img[idx].unsqueeze(axis=0)).to('float32'))
            #     temp_len += im_len
            # if temp_len > self.max_length:
            #     break
        # wrap_tokens = paddle.concat(x=wrap_tokens, axis=1)
        # wrap_embeds = paddle.concat(x=wrap_embeds, axis=1)
        # wrap_atts = paddle.concat(x=wrap_atts, axis=1)
        # wrap_im_mask = paddle.concat(x=wrap_im_mask, axis=1)
        # wrap_target = self.mask_human_targets(wrap_tokens)
        # wrap_embeds = wrap_embeds[:, :self.max_length]
        # wrap_atts = wrap_atts[:, :self.max_length]
        # wrap_target = wrap_target[:, :self.max_length]
        # wrap_im_mask = wrap_im_mask[:, :self.max_length]
        #     wrap_embeds_list.append(wrap_embeds)
        #     wrap_atts_list.append(wrap_atts)
        #     wrap_target_list.append(wrap_target)
        #     wrap_im_mask_list.append(wrap_im_mask)
        # wrap_embeds = paddle.concat(x=wrap_embeds_list)
        # wrap_atts = paddle.concat(x=wrap_atts_list)
        # wrap_target = paddle.concat(x=wrap_target_list)
        # wrap_im_mask = paddle.concat(x=wrap_im_mask_list)
        # return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask
        return wrap_tokens, text

    def text2emb(self, text, add_special=False):
        to_regress_tokens = self.tokenizer(text, return_tensors='pd',
            padding='longest', truncation=True, max_length=self.max_length,
            add_special_tokens=add_special)
        targets = self.mask_human_targets(to_regress_tokens.input_ids)
        return to_regress_tokens, targets

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    if end_count % 2 == 0:
                        targets[last_eoa:i + 6] = -100
                    else:
                        last_eoa = i + 1
                    end_count += 1
                elif temp_id == 2:
                    targets[i + 1:] = -100
                    break
            if temp_id != 2 and end_count % 2 == 0:
                targets[last_eoa + 1:] = -100
            target_batch.append(targets.unsqueeze(axis=0))
        target_batch = paddle.concat(x=target_batch, axis=0)
        return target_batch


    def batch_decode(self, *args, **kwargs):

        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """

        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, pred: Union[List, paddle.Tensor]):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """

        return self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class InternLMXComposer2ImageProcessor(BaseImageProcessor):
    def __init__(self, image_size=224, **kwargs):
        super().__init__(**kwargs)
        mean = 0.48145466, 0.4578275, 0.40821073
        std = 0.26862954, 0.26130258, 0.27577711
        self.normalize = transforms.Normalize(mean, std)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation='bicubic'),
            transforms.ToTensor(), self.normalize])

    def __call__(self, image_paths: List[str]):

        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))

        images = paddle.stack(x=images, axis=0)
        return images


class InternLMXComposer2TextProcessor(BaseTextProcessor):

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def __call__(self, sources):
        END_HUMAN = '[UNUSED_TOKEN_145]\n'
        END_BOT = '[UNUSED_TOKEN_145]\n'
        # import pdb
        # pdb.set_trace()
        # conversation = ''
        # for idx, sentence in enumerate(sources):
        #     BEGIN_SIGNAL = ''
        #     from_str = sentence['from']
        #     if from_str.lower() == 'human' or from_str.lower() == 'user':
        #         from_str = '[UNUSED_TOKEN_146]user\n'
        #         temp = BEGIN_SIGNAL + from_str + sentence['value'].strip(
        #             ) + END_HUMAN
        #     else:
        #         from_str = '[UNUSED_TOKEN_146]assistant\n'
        #         temp = BEGIN_SIGNAL + from_str + sentence['value'].strip(
        #             ) + END_BOT
        #     conversation += temp
        conversation = '[UNUSED_TOKEN_146]user\n' + sources[0].strip() + END_HUMAN + \
                       '[UNUSED_TOKEN_146]assistant\n' + sources[1].strip() + END_BOT + '</s>'
        return conversation
