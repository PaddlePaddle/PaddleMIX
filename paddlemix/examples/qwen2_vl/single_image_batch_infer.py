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

from paddlemix.models.qwen2_vl import MIXQwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

image_processor = Qwen2VLImageProcessor()
tokenizer = MIXQwen2Tokenizer.from_pretrained(MODEL_NAME)
processor = Qwen2VLProcessor(image_processor, tokenizer)

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

image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"

question2 = messages2[0]["content"][1]["text"]
text2 = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question2}<|im_end|>\n<|im_start|>assistant\n"

question3 = messages3[0]["content"][1]["text"]
text3 = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question3}<|im_end|>\n<|im_start|>assistant\n"

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
