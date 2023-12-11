# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlemix import QWenLMHeadModel, QWenTokenizer, QwenVLProcessor
from paddlemix.utils.log import logger

paddle.seed(1234)
dtype = "bfloat16"
if not paddle.amp.is_bfloat16_supported():
    logger.warning("bfloat16 is not supported on your device,change to float32")
    dtype = "float32"

tokenizer = QWenTokenizer.from_pretrained("qwen-vl/qwen-vl-chat-7b", dtype=dtype)
processor = QwenVLProcessor(tokenizer=tokenizer)
model = QWenLMHeadModel.from_pretrained("qwen-vl/qwen-vl-chat-7b", dtype=dtype)
model.eval()

# 第一轮对话
query1 = [
    {"image": "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"},
    {"text": "这是什么?"},
]

input = processor(query=query1, return_tensors="pd")
query1 = tokenizer.from_list_format(query1)
response, history = model.chat(tokenizer, query=query1, history=None, images=input["images"])

print("answer1:", response)
# 这张图片展示了一辆红色的 Beacon Bus 正在行驶，它在道路上与其它车辆共同行驶。
# bus 上的数字显示它正在前往特定地点，可能是一个公共汽车 stop。
# 在场景中还可以看到一辆汽车和另一辆巴士，它们位于不同的位置上。人们周围走动，其中一些人甚至在斑马线上行走。
# 这场景描绘了繁忙的交通和运输在城市中的重要性。

# 第二轮对话
query2 = "框出图中公交车的位置"
response, history = model.chat(tokenizer, query2, history=history)
print("answer2:", response)
# <ref>公交车</ref><box>(178,279),(806,884)</box>
