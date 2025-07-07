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

import logging

import paddle
import paddle.nn as nn
from paddlenlp.transformers import AutoTokenizer, T5Config, T5EncoderModel


class FlanT5HiddenState(nn.Layer):
    """
    llama = FlanT5HiddenState()
    data = ["","this is not an empty sentence"]
    encoder_hidden_states = llama(data)
    import ipdb;ipdb.set_trace()
    """

    def __init__(
        self, text_encoder_name="t5-v1_1-large", freeze_text_encoder=True  # t5-v1_1-large -> google/flan-t5-large
    ):
        super().__init__()
        self.freeze_text_encoder = freeze_text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.model = T5EncoderModel(T5Config.from_pretrained(text_encoder_name))
        if freeze_text_encoder:
            self.model.eval()
            for p in self.model.parameters():
                p.stop_gradient = True
        else:
            print("=> The text encoder is learnable")

        self.empty_hidden_state_cfg = None
        self.device = None

    # Required
    def get_unconditional_condition(self, batchsize):
        param = self.model.parameters()[0]
        if self.freeze_text_encoder:
            assert param.stop_gradient

        # device = param.device
        if self.empty_hidden_state_cfg is None:
            self.empty_hidden_state_cfg, _ = self([""])

        hidden_state = paddle.cast(paddle.concat([self.empty_hidden_state_cfg] * batchsize), dtype="float32")
        attention_mask = paddle.ones((batchsize, hidden_state.shape[1]), dtype="float32")
        return [hidden_state, attention_mask]  # Need to return float type

    def forward(self, batch):
        param = self.model.parameters()[0]
        if self.freeze_text_encoder:
            assert param.stop_gradient

        try:
            return self.encode_text(batch)
        except Exception as e:
            print(e, batch)
            logging.exception("An error occurred: %s", str(e))

    def encode_text(self, prompt):
        # device = self.model.device
        batch = self.tokenizer(
            prompt,
            max_length=128,  # self.tokenizer.model_max_length
            padding=True,
            truncation=True,
            return_tensors="pd",
        )
        input_ids, attention_mask = batch.input_ids, batch.attention_mask
        # Get text encoding
        if self.freeze_text_encoder:
            with paddle.no_grad():
                encoder_hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        else:
            encoder_hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        return [
            encoder_hidden_states.detach(),
            paddle.cast(attention_mask, dtype="float32"),
        ]  # Attention mask == 1 means usable token
