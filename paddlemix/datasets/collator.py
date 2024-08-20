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
                    if not isinstance(raw_data["images"][0], list):
                        raw_data["images"] = [raw_data["images"]]
                    raw_data["images"] = [self.processor.image_processor(path) for path in raw_data["images"]]
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


class InternLMXComposer2Collator:
    """Collate examples for InternLMXComposer2Collator"""

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, instances):

        instances = [self.processor(query=instance, mode=self.mode) for instance in instances]

        input_tokens, input_text = tuple(
            [instance[key] for instance in instances] for key in ("input_tokens", "input_text")
        )
        batch = dict(
            input_tokens=input_tokens,
            input_text=input_text,
        )
        if "images" in instances[0].keys():
            input_images = tuple([instance["images"] for instance in instances])
            batch["images"] = input_images

        return dict(samples=batch)


class InternVL2Collator:
    """Collate examples for InternVL2Collator"""

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, features):
        pad_id = self.processor.tokenizer.pad_token_id
        IGNORE_INDEX = -100
        first = features[0]
        batch = {}

        batch_lens = [feat["input_ids"].shape for feat in features]
        max_item_length = max(batch_lens)[0]
        for idx in range(len(features)):
            feat = self.processor(features[idx])
            temp_input_ids = paddle.to_tensor([pad_id] * max_item_length, dtype=paddle.int64)
            temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
            feat["input_ids"] = temp_input_ids
            temp_labels = paddle.to_tensor([IGNORE_INDEX] * max_item_length, dtype=paddle.int64)
            temp_labels[: feat["labels"].shape[0]] = feat["labels"]
            feat["labels"] = temp_labels
            feat["attention_mask"] = feat["input_ids"].ne(pad_id)

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], paddle.Tensor) else first["label"]
            dtype = paddle.int64 if isinstance(label, int) else paddle.float32
            batch["labels"] = paddle.to_tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], paddle.Tensor):
                batch["labels"] = paddle.stack([f["label_ids"] for f in features])
            else:
                dtype = paddle.int64 if isinstance(first["label_ids"][0], int) else paddle.float32
                batch["labels"] = paddle.to_tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if (
                k not in ("label", "label_ids", "pixel_values", "image_flags")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, paddle.Tensor):
                    batch[k] = paddle.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = paddle.to_tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = paddle.to_tensor([f[k] for f in features])
            if k in ("pixel_values", "image_flags"):
                if isinstance(v, paddle.Tensor):
                    batch[k] = paddle.concat([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = paddle.concat(np.stack([f[k] for f in features]))
                else:
                    batch[k] = paddle.concat([f[k] for f in features])
        return batch
