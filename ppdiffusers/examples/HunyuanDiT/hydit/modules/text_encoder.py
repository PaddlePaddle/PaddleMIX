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

import paddle
import paddle.nn as nn

from ppdiffusers.transformers import (
    AutoTokenizer,
    T5EncoderModel,
    T5ForConditionalGeneration,
)


class MT5Embedder(nn.Layer):
    available_models = ["t5-v1_1-xxl"]

    def __init__(
        self,
        model_dir="t5-v1_1-xxl",
        model_kwargs=None,
        paddle_dtype=None,
        use_tokenizer_only=False,
        conditional_generation=False,
        max_length=128,
    ):
        super().__init__()
        self.paddle_dtype = paddle_dtype or paddle.bfloat16
        self.max_length = max_length
        if model_kwargs is None:
            model_kwargs = {
                # "low_cpu_mem_usage": True,
                # "paddle_dtype": self.paddle_dtype,
            }
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if use_tokenizer_only:
            return
        if conditional_generation:
            self.model = None
            self.generation_model = T5ForConditionalGeneration.from_pretrained(model_dir)
            return
        self.model = T5EncoderModel.from_pretrained(model_dir, **model_kwargs).eval().to(dtype=self.paddle_dtype)

    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pd",
        )
        tokens = text_tokens_and_mask["input_ids"][0]
        mask = text_tokens_and_mask["attention_mask"][0]
        # tokens = torch.tensor(tokens).clone().detach()
        # mask = torch.tensor(mask, dtype=torch.bool).clone().detach()
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pd",
        )

        with paddle.no_grad():
            outputs = self.model(
                input_ids=text_tokens_and_mask["input_ids"],
                attention_mask=text_tokens_and_mask["attention_mask"] if attention_mask else None,
                output_hidden_states=True,
            )
            text_encoder_embs = outputs["hidden_states"][layer_index].detach()

        return text_encoder_embs, text_tokens_and_mask["attention_mask"]

    @paddle.no_grad()
    def __call__(self, tokens, attention_mask, layer_index=-1):
        with paddle.amp.autocast():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        z = outputs.hidden_states[layer_index].detach()
        return z

    def general(self, text: str):
        # input_ids = input_ids = torch.tensor([list(text.encode("utf-8"))]) + num_special_tokens
        input_ids = self.tokenizer(text, max_length=128).input_ids
        print(input_ids)
        outputs = self.generation_model(input_ids)
        return outputs
