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

IGNORE_INDEX = -100


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = paddle.to_tensor([pad_id] * max_item_length, dtype="int64")
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = paddle.to_tensor([IGNORE_INDEX] * max_item_length, dtype="int64")
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].not_equal(paddle.to_tensor(pad_id))  # .ne .not_equal

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
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, paddle.Tensor):
                batch[k] = paddle.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = paddle.to_tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = paddle.to_tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = paddle.to_tensor([pad_id] * max_item_length, dtype="int64")
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = paddle.to_tensor([IGNORE_INDEX] * max_item_length, dtype="int64")
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].not_equal(paddle.to_tensor(pad_id))

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
        if k not in ("label", "label_ids", "pixel_values", "image_flags") and v is not None and not isinstance(v, str):
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
