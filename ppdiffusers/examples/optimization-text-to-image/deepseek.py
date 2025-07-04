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

from openai import OpenAI

client = OpenAI(api_key="xxxx", base_url="https://api.deepseek.com/beta")

import argparse

parser = argparse.ArgumentParser(description="参数")
parser.add_argument("prompt", type=str, help="图片描述")
args = parser.parse_args()
prompt = args.prompt

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {
            "role": "system",
            "content": "You are a picture beautifier. The user will provide you with a description of a picture, and you need to beautify this description from multiple aspects and output the refined picture description, which will be used for text - to - image generation. You only need to output the final prompt!",
        },
        {"role": "user", "content": prompt},
    ],
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
print(reasoning_content)
print("=================================================")
print(content)
