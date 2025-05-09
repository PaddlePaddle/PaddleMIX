# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# code is heavily based on https://github.com/tianweiy/DMD2

import lmdb
import numpy as np
from paddle.io import Dataset
from utils import get_array_shape_from_lmdb, retrieve_row_from_lmdb


class SDImageDatasetLMDB(Dataset):
    def __init__(self, dataset_path, tokenizer_one, is_sdxl=False, tokenizer_two=None):
        self.KEY_TO_TYPE = {"latents": np.float16}
        self.is_sdxl = is_sdxl  # sdxl uses two tokenizers
        self.dataset_path = dataset_path
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.latent_shape = get_array_shape_from_lmdb(self.env, "latents")

        self.length = self.latent_shape[0]

        print(f"Dataset length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = retrieve_row_from_lmdb(self.env, "latents", self.KEY_TO_TYPE["latents"], self.latent_shape[1:], idx)
        image = image.astype(np.float32)

        with self.env.begin() as txn:
            prompt = txn.get(f"prompts_{idx}_data".encode()).decode()

        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pd",
        ).input_ids

        output_dict = {
            "images": image,
            "text_input_ids_one": text_input_ids_one,
        }

        if self.is_sdxl:
            text_input_ids_two = self.tokenizer_two(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pd",
            ).input_ids
            output_dict["text_input_ids_two"] = text_input_ids_two

        return output_dict
