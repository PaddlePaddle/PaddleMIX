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

from paddlemix import QWenLMHeadModel, QWenTokenizer
from paddlemix.utils.log import logger

paddle.seed(1234)
dtype = "bfloat16"
if not paddle.amp.is_bfloat16_supported():
    logger.warning("bfloat16 is not supported on your device,change to float32")
    dtype = "float32"

tokenizer = QWenTokenizer.from_pretrained("qwen-vl/qwen-vl-chat-7b", dtype=dtype)

model = QWenLMHeadModel.from_pretrained("qwen-vl/qwen-vl-chat-7b", dtype=dtype)
model.eval()

# 第一轮对话
query1 = tokenizer.from_list_format(
    [
        {
            "image": "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
        },  # Either a local path or an url
        {"text": "这是什么?"},
    ]
)


response, history = model.chat(tokenizer, query=query1, history=None)
print("answer1:", response)
# 这张图片展示了一辆红色的 Beacon Bus 正在行驶，它是一辆公共汽车。车身上有黑色的字母，表明其服务线路。
# 在公共汽车上，乘客可以使用数字板获取信息和目的地。公共汽车前方是一辆银色的汽车，它们都正在行驶在道路上。
# 另外，还有一些人可以被看到在公共汽车附近。在背景中，一个标志牌也被看到

# 第二轮对话
query2 = "框出图中公交车的位置"
response, history = model.chat(tokenizer, query2, history=history)
print("answer2:", response)
# <ref>公交车</ref><box>(178,280),(802,894)</box>
