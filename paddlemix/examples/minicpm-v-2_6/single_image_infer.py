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

# from paddlenlp.transformers import AutoTokenizer
from PIL import Image

from paddlemix.models.minicpm_v.modeling_minicpmv import MiniCPMV
from paddlemix.models.minicpm_v.tokenization_minicpmv_fast import MiniCPMVTokenizerFast

MODEL_NAME = "openbmb/MiniCPM-V-2_6"
model = MiniCPMV.from_pretrained(MODEL_NAME, dtype="bfloat16")
model = model.eval()
tokenizer = MiniCPMVTokenizerFast.from_pretrained(MODEL_NAME)
image = Image.open("paddlemix/demo_images/minicpm_demo.jpg").convert("RGB")

question = "识别图片中的手写文字。"

msgs = [{"role": "user", "content": [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    max_new_tokens=2048,  # 2048
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    stream=False,
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end="")
