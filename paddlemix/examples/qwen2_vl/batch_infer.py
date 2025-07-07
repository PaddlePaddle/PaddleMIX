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


from paddlemix.models.qwen2_vl import MIXQwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", dtype="bfloat16")

image_processor = Qwen2VLImageProcessor()
tokenizer = MIXQwen2Tokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "paddlemix/demo_images/examples_image1.jpg"},
            {"type": "image", "image": "paddlemix/demo_images/examples_image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]

messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]


messages3 = [
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

# Combine messages for batch processing
messages = [messages1, messages3]

# Preparation for batch inference
texts = [processor.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
import pdb

pdb.set_trace()
# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
for i in range(len(output_text)):
    print(f"Generated output for message {i}: {output_text[i]}")
