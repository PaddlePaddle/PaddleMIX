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

pretrained = "lmms-lab/llava-critic-7b"
# pretrained = "lmms-lab/llava-onevision-qwen2-7b-chat"

model = LlavaQwenForCausalLM.from_pretrained(pretrained, dtype=paddle.bfloat16).eval()
tokenizer = Qwen2Tokenizer.from_pretrained(pretrained)
image_processor = SigLipImageProcessor()

image = Image.open("paddlemix/demo_images/critic_img_seven.png")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.cast(paddle.bfloat16) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

# pairwise ranking
critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe first response: [The image is a black and white sketch of a line that appears to be in the shape of a cross. The line is a simple and straightforward representation of the cross shape, with two straight lines intersecting at a point.]\nThe second response: [This is a handwritten number seven.]\nASSISTANT:\n"

# pointwise scoring
# critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of answer answers provided by a Large Multimodal Model (LMM). Score the response out of 100 and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe LMM response: [This is a handwritten number seven.]\nASSISTANT:\n "

question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
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
