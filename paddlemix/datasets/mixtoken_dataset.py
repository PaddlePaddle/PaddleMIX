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
from paddle.io import Dataset
from scipy.linalg import block_diag
from tqdm import tqdm


class MIXToken:
    required_input_keys = ["input_ids", "labels"]
    required_output_keys = ["input_ids", "labels", "attention_mask"]
    # Only supported the following keys for MIXToken. Keys outside of the set will be ignored.
    supported_input_keys = ["input_ids", "labels", "attention_mask", "position_ids", "images"]

    @classmethod
    def _pad_batch_records(cls, batch_records):
        # Only consider supported input keys
        input_keys = [key for key in batch_records[0].keys() if key in cls.supported_input_keys]

        # Check required_keys
        for key in cls.required_input_keys:
            if key not in input_keys:
                raise ValueError(f"feature `{key}` is required for MIXTokenDataset")
        # Output features must include all required output keys
        for key in cls.required_output_keys:
            if key not in input_keys:
                input_keys.append(key)

        batched_features = {key: [] for key in input_keys}

        for record in batch_records:
            batched_features["input_ids"].extend(record["input_ids"])
            batched_features["labels"].extend(record["labels"])
            seq_length = len(record["input_ids"])
            # If attention_mask is not given, assume it's causal mask
            attention_mask = record.get("attention_mask", np.tril(np.ones([seq_length, seq_length], dtype=bool)))
            batched_features["attention_mask"].append(attention_mask)
            # NOTE: position_ids is optional and not required by every model
            # We append instead of extend here to accommodate 2D position ids
            if "position_ids" in record:
                batched_features["position_ids"].append(record["position_ids"])
            if "images" in record:
                batched_features["images"].append(record["images"])

        block_attention_mask = block_diag(*batched_features["attention_mask"])
        # convert to 3-D [batch_size(1), seq_length, seq_length]
        batched_features["attention_mask"] = np.expand_dims(block_attention_mask, axis=0)
        if "position_ids" in batched_features:
            # Accommodate both 1D and 2D position ids
            batched_features["position_ids"] = np.concatenate(batched_features["position_ids"], axis=-1).tolist()
        return batched_features


class MIXTokenMapDataset(MIXToken, Dataset):
    """
    MIXToken is a unique feature of PaddleMix training, which replaces traditional pad tokens by
    concatenating effective tokens to increase the throughput of a single sample and improve training speed.

    traditional pad tokens:
    len( imageToken + query + paddingToken ) = max_length

    MIXToken:
    len( imageToken1 + query1 + imageToken2 + query2 + ... + paddingToken ) = max_length

    """

    def __init__(self, data, max_length, processor=None, tokenizer=None, mode="train"):
        self.max_length = max_length
        self.processor = processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.new_data = self._create_intokens_data(data)

    def _create_intokens_data(self, data):
        batch_records, max_len = [], 0
        cur_len_so_far = 0

        total_data = []

        for i in tqdm(range(len(data))):
            record = data[i]

            if self.processor:
                record = self.processor(record=record, mode=self.mode)

            if getattr(self.tokenizer, "image_token_span", None) is not None and record["images"] is not None:
                image_token_span = self.tokenizer.image_token_span - 1  # image token
            else:
                image_token_span = 0

            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + int(image_token_span) + len(record["input_ids"])) <= self.max_length

            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"]) + image_token_span
            else:
                # exceed max length
                padded_list = self._pad_batch_records(batch_records)
                total_data.append(padded_list)
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"]) + image_token_span

        # remaining data
        if batch_records:
            padded_list = self._pad_batch_records(batch_records)
            total_data.append(padded_list)

        return total_data

    def __getitem__(self, idx):
        return self.new_data[idx]

    def __len__(self):
        return len(self.new_data)
