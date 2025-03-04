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

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

image_processor = Qwen2_5_VLImageProcessor()
tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(MODEL_NAME)

processor = Qwen2_5_VLProcessor(image_processor, tokenizer)

# min_pixels = 256*28*28 # 200704
# max_pixels = 1280*28*28 # 1003520
# processor = Qwen2_5_VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "paddlemix/demo_images/examples_image1.jpg"},
            {"type": "image", "image": "paddlemix/demo_images/examples_image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]

# Preparation for inference
image_inputs, video_inputs = process_vision_info(messages)

question = "Identify the similarities between these images."
texts = [processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)  # already trimmed in paddle
output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("output_text:\n", output_text[0])
