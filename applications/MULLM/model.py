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

import numpy as np
import paddle
from PIL import Image

from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from paddlemix.processors.qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
    process_vision_info,
)


class ImageChatModel:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize the model and processors.

        Args:
            model_path: str, path to the model
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype="bfloat16", attn_implementation="eager"
        )

        self.image_processor = Qwen2_5_VLImageProcessor()
        self.tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(model_path)
        min_pixels = 256 * 28 * 28  # 200704
        max_pixels = 1280 * 28 * 28  # 1003520
        self.processor = Qwen2_5_VLProcessor(
            self.image_processor, self.tokenizer, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def generate_description(self, image: np.ndarray, question: str) -> str:
        """
        Generate text description for an image based on a question.

        Args:
            image: numpy array
            question: str, the question about the image

        Returns:
            str: Generated text description
        """
        # Prepare messages
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        texts = [self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        yield "请稍等，正在分析图片..."
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )

        # Generate response
        with paddle.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.01)
            output_text = self.processor.batch_decode(
                generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        yield output_text[0]
