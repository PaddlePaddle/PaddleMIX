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

from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from paddlemix.processors.qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
    process_vision_info,
)
import paddle

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
compute_dtype = "bfloat16"
paddle.set_default_dtype(compute_dtype)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=compute_dtype, attn_implementation="flash_attention_2")

image_processor = Qwen2_5_VLImageProcessor()
tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(MODEL_NAME)

processor = Qwen2_5_VLProcessor(image_processor, tokenizer)

# min_pixels = 256*28*28 # 200704
# max_pixels = 1280*28*28 # 1003520
# processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "paddlemix/demo_images/twitter3.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "paddlemix/demo_images/examples_image2.jpg",
            },
            {"type": "text", "text": "What is the animal in this image?"},
        ],
    }
]

messages = [messages2, messages3]

# Preparation for inference
image_inputs, video_inputs = process_vision_info(messages)

text2 = processor.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
text3 = processor.tokenizer.apply_chat_template(messages3, tokenize=False, add_generation_prompt=True)

text = [text2, text3]

inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)  # already trimmed in paddle
output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("output_text:\n", output_text)  # list
