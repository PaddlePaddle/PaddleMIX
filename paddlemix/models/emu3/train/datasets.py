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

import json
import os
import random

import paddle


class Emu3FeatureDataset(paddle.io.Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        with open(args.data_path) as f:
            d = json.load(f)
        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"]
        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        path = os.path.join(self.path_prefix, self.filelist[index])
        data = paddle.load(path=str(path))
        image_tokens = data["images"]
        image_prompt = self.format_image_prompt(image_tokens)
        p_prob = random.random()
        if p_prob < self.args.null_prompt_prob:
            prompt = ""
        else:
            prompt = data["texts"]
        input = self.tokenizer.bos_token + prompt + image_prompt
        sample = self.tokenizer(input, padding="max_length", return_token_type_ids=False, return_tensors="pt")
        labels = sample["input_ids"]
        if self.args.apply_loss_on_only_vision:
            labels = paddle.where(
                condition=paddle.logical_and(x=labels >= self.bov, y=labels <= self.eov),
                x=labels,
                y=self.args.ignore_index,
            )
        sample["labels"] = labels
        for k, v in sample.items():
            sample[k] = v.squeeze(axis=0)
        return sample

    def format_image_prompt(self, image_tokens):
        h, w = tuple(image_tokens.shape)
        imgstr = self.to_imgstr(image_tokens)
        image_prompt = (
            self.tokenizer.boi_token
            + f"{h}*{w}"
            + self.tokenizer.img_token
            + imgstr
            + self.tokenizer.eol_token
            + self.tokenizer.eof_token
            + self.tokenizer.eoi_token
        )
        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [self.args.visual_token_pattern.format(token_id=token_id) for token_id in token_row]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr
