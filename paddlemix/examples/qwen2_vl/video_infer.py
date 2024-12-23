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

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

image_processor = Qwen2VLImageProcessor()
tokenizer = MIXQwen2Tokenizer.from_pretrained(MODEL_NAME)
min_pixels = 256 * 28 * 28  # 200704
max_pixels = 1280 * 28 * 28  # 1003520
processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

# Messages containing a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "paddlemix/demo_images/red-panda.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

image_inputs, video_inputs = process_vision_info(messages)
question = "Describe this video."
video_pad_token = "<|vision_start|><|video_pad|><|vision_end|>"
text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{video_pad_token}{question}<|im_end|>\n<|im_start|>assistant\n"

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)  # already trimmed in paddle
# print("generated_ids:\n", generated_ids)
output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("output_text:\n", output_text[0])
