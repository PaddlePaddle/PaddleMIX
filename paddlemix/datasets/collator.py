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

import numpy as np
import paddle


class CLIPCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data_list):
        if isinstance(data_list[0], dict):
            images = [sample["image"] for sample in data_list]
            text = [sample["text_input"] for sample in data_list]
            batch = self.processor(
                images=images,
                text=text,
                max_length=77,
                return_tensors="pd",
                return_attention_mask=False,
                mode="train",
                padding_zero=True,
            )
            return batch
        else:
            images = [sample[0] for sample in data_list]
            labels = [sample[1] for sample in data_list]
            batch = self.processor(
                images=images,
                text=None,
                max_length=77,
                return_tensors="pd",
                return_attention_mask=False,
                mode="eval",
                do_resize=True,
                do_crop=True,
                padding_zero=True,
            )
            batch["labels"] = paddle.to_tensor(np.array(labels))
            return batch


class EVA02Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample[0] for sample in data_list]
        # get labels from teacher's clip_features
        batch = self.processor(
            images=images,
            return_tensors="pd",
            mode=self.mode,
        )
        return batch


class MiniGPT4Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        target_texts = [sample["text_input"] for sample in data_list]
        # random text from text_list read by processor and combine it with default prompt
        batch_data = self.processor(images=images, mode="train")
        target_outputs = self.processor.process_target_texts(target_texts)
        batch_data.update(target_outputs)
        return batch_data


class QwenVLCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):

        input_ids = []
        labels = []
        images = []
        IGNORE_TOKEN_ID = -100
        for record in data_list:

            if isinstance(record, dict) and "input_ids" in record.keys():
                raw_data = record
            else:
                raw_data = self.processor(query=record, mode=self.mode)

            raw_data["input_ids"] += [self.processor.tokenizer.pad_token_id] * (
                self.processor.max_len - len(raw_data["input_ids"])
            )
            raw_data["labels"] += [IGNORE_TOKEN_ID] * (self.processor.max_len - len(raw_data["labels"]))
            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

            if "images" in raw_data:

                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)

                images.append(raw_data["images"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int32")
        labels = paddle.to_tensor(data=labels, dtype="int32")
        attention_mask = input_ids.not_equal(y=paddle.to_tensor(self.processor.tokenizer.pad_token_id, dtype="int32"))

        if len(images) > 0:
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(
            input_ids=input_ids,
            labels=labels,
            images=images if 0 < len(images) else None,
            attention_mask=attention_mask,
        )

        return batch_data


class VisualglmCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test", max_seq_length=2048):
        self.processor = processor
        self.mode = mode
        self.max_seq_length = max_seq_length

    def __call__(self, data_list):

        input_ids = []
        labels = []
        images = []

        for record in data_list:
            if "input_ids" not in record.keys():
                raw_data = self.processor(record=record, mode=self.mode)
            else:
                raw_data = record

            pad_len = self.max_seq_length - len(raw_data["input_ids"])
            raw_data["input_ids"] = raw_data["input_ids"] + [self.processor.tokenizer.pad_token_id] * pad_len
            raw_data["labels"] = raw_data["labels"] + [self.processor.tokenizer.pad_token_id] * pad_len
            raw_data["labels"] = [
                (l if l != self.processor.tokenizer.pad_token_id else -100) for l in raw_data["labels"]
            ]

            if "images" in raw_data:
                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)
                images.append(raw_data["images"])

            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int64")
        labels = paddle.to_tensor(data=labels, dtype="int64")

        if 0 < len(images):
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(input_ids=input_ids, labels=labels, images=images if 0 < len(images) else None)
        return batch_data


class LLaVACollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test", mixtokens=False):
        self.processor = processor
        self.mode = mode
        self.mixtokens = mixtokens

    def __call__(self, data_list):
        IGNORE_INDEX = -100
        input_ids = []
        labels = []
        images = []
        for record in data_list:

            if isinstance(record, dict) and "input_ids" in record.keys():
                raw_data = record
            else:
                raw_data = self.processor(record=record, mode=self.mode)

            raw_data["input_ids"] += [self.processor.tokenizer.pad_token_id] * (
                self.processor.max_len - len(raw_data["input_ids"])
            )
            raw_data["labels"] += [IGNORE_INDEX] * (self.processor.max_len - len(raw_data["labels"]))

            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

            if "images" in raw_data:
                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)

                images.append(raw_data["images"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int32")
        labels = paddle.to_tensor(data=labels, dtype="int32")
        attention_mask = input_ids.not_equal(y=paddle.to_tensor(self.processor.tokenizer.pad_token_id, dtype="int32"))

        if len(images) > 0:
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(
            input_ids=input_ids,
            labels=labels,
            images=images if len(images) > 0 else None,
            attention_mask=attention_mask,
        )

        return batch_data
