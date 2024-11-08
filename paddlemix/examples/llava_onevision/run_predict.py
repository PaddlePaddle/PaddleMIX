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

import copy

import paddle
from paddlenlp.transformers import Qwen2Tokenizer
from PIL import Image

from paddlemix.models.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.llava.language_model.llava_qwen import LlavaQwenForCausalLM
from paddlemix.models.llava.mm_utils import process_images
from paddlemix.models.llava.multimodal_encoder.siglip_encoder import (
    SigLipImageProcessor,
)
from paddlemix.models.llava.train_utils import tokenizer_image_token

pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
# pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# pretrained = "lmms-lab/llava-onevision-qwen2-7b-si"
# pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"

model = LlavaQwenForCausalLM.from_pretrained(pretrained, dtype=paddle.bfloat16).eval()
tokenizer = Qwen2Tokenizer.from_pretrained(pretrained)
image_processor = SigLipImageProcessor()

image = Image.open("paddlemix/demo_images/llava_v1_5_radar.jpg")

image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.cast(paddle.bfloat16) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

prompt = "What is shown in this image?"

question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pd").unsqueeze(0)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont[0], skip_special_tokens=True)
print("output:\n", text_outputs[0])
