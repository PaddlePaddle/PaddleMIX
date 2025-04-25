# -*- coding: utf-8 -*-

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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)

from paddlenlp.transformers import CLIPImageProcessor, Qwen2Tokenizer
from PIL import Image

from paddlemix.models.points_qwen2_5 import POINTSChatModel

model_path = "WePOINTS/POINTS-Qwen-2-5-7B-Chat"

model = POINTSChatModel.from_pretrained(model_path)
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
image_processor = CLIPImageProcessor.from_pretrained(model_path)

image_path = "paddlemix/demo_images/examples_image2.jpg"
pil_image = Image.open(image_path)
prompt = "please describe the image in detail"

generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 0.0,
    "num_beams": 1,
}
res = model.chat(pil_image, prompt, tokenizer, image_processor, True, generation_config)

print(res)
