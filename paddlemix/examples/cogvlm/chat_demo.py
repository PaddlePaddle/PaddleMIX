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

import random

import numpy as np
import paddle

seed = 2024
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

import argparse

from paddlenlp.transformers import AutoTokenizer
from PIL import Image

from paddlemix.models.cogvlm.modeling import CogModelForCausalLM

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path", type=str, default="THUDM/cogagent-chat", help="pretrained ckpt and tokenizer"
)

args = parser.parse_args()
MODEL_PATH = args.model_name_or_path
TOKENIZER_PATH = MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

data_type = "float32"

model = CogModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=data_type,
    low_cpu_mem_usage=False,
)
model.eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
while True:
    image_path = input("image path >>>>> ")
    if image_path == "":
        print("You did not enter image path, the following will be a plain text conversation.")
        image = None
        text_only_first_query = True
    else:
        image = Image.open(image_path).convert("RGB")
    history = []
    while True:
        query = input("Human:")
        if query == "clear":
            break
        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ""
                for _, (old_query, response) in enumerate(history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)
        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, template_version="base"
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, images=[image]
            )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(axis=0),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(axis=0),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(axis=0),
            "images": [[input_by_model["images"][0].cast(data_type)]] if image is not None else None,
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [[input_by_model["cross_images"][0].cast(data_type)]]
        gen_kwargs = {"max_new_tokens": 2048, "do_sample": False}
        with paddle.no_grad():
            outputs, _ = model.generate(**inputs, **gen_kwargs)
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
        history.append((query, response))
