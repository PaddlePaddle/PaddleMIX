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

import paddle
import paddle.nn.functional as F
from paddlemix.models.qwen2_vl import MIXQwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)
from PIL import Image
import requests
from typing import List, Dict, Optional


class GmeQwen2VL:
    """
    GmeQwen2VL-Qwen2-VL provides computation of text, image, and multimodal embeddings.
    """
    def __init__(
        self,
        model_name: str = "GME-Qwen2-VL/gme-Qwen2-VL-2B-Instruct",
        model_path: Optional[str] = None,
        device: str = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu",
        max_length: int = 1800,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        model_name = model_path or model_name  # Use local model if path is provided
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, dtype="bfloat16")
        self.model.eval()  # Set to inference mode
        self.model.to(device)  # Move to specified device

        self.max_length = max_length
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28

        self.image_processor = Qwen2VLImageProcessor(min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        self.tokenizer = MIXQwen2Tokenizer.from_pretrained(model_name)
        self.processor = Qwen2VLProcessor(self.image_processor, self.tokenizer)

        self.device = device
        
        self.normalize = normalize
        self.default_instruction = "You are a helpful assistant."  # Default instruction


    def embed(
        self, 
        texts: List[str], 
        images: List[Image.Image] = None, 
        instruction: Optional[str] = None, 
        is_query: bool = True,
    ) -> paddle.Tensor:
        """
        Compute embeddings for text and images, ensuring correct batch concatenation logic.
        """

        # Select appropriate instruction
        if not is_query or instruction is None:
            instruction = self.default_instruction

        input_texts, input_images = [], []
        
        # Process text & images
        has_text = any(texts)  # Check if text is included
        has_image = images is not None and any(images)  # Check if images are included

        for text, image in zip(texts, images or [None] * len(texts)):
            input_str = ""

            # Process image
            if image is not None:
                input_str += "<|vision_start|><|image_pad|><|vision_end|>"
                input_images.append(image)  # Collect processed images
            else:
                input_images.append(None)  # Ensure image alignment

            # Process text
            if text is not None:
                input_str += text  # Append text content

            formatted_text = f'<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
            input_texts.append(formatted_text)  # Store final formatted text
        

        # Process image information
        if has_image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": ""}  
                    ],
                }
                for img in images
            ]
            image_inputs, _ = process_vision_info(messages)
        else:
            image_inputs = None 

        # Process Tokenization
        inputs = self.processor(
            text=input_texts,
            images=image_inputs,
            padding=True,
            max_length=self.max_length,
            return_tensors="pd",
        )

        # Compute embeddings
        with paddle.no_grad():
            embeddings = self.model.gme_qwen2_vl_forward(**inputs)

        # Normalize embeddings
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, axis=1)

        return  embeddings


    def get_text_embeddings(self, texts: List[str], **kwargs) -> paddle.Tensor:
        """
        Compute text embeddings
        """
        return self.embed(texts=texts, images=None, **kwargs)

    def get_image_embeddings(self, images: List[Image.Image], **kwargs) -> paddle.Tensor:
        """
        Compute image embeddings
        """
        return self.embed(texts=[""] * len(images), images=images, **kwargs)

    def get_fused_embeddings(self, texts: List[str] = None, images: List[Image.Image] = None, **kwargs) -> paddle.Tensor:
        """
        Compute fused embeddings for text+image
        Supports:
        - Text only
        - Image only
        - Text+image fusion
        """
        return self.embed(texts=texts, images=images, **kwargs)


    def encode_queries(self, queries: List[str], **kwargs) -> paddle.Tensor:
        """
        Compute embeddings for query texts, typically used for search tasks.
        """
        return self.get_text_embeddings(texts=queries, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> paddle.Tensor:
        """
        Compute embeddings for corpus texts, typically used for indexing databases.
        """
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.get_text_embeddings(texts=sentences, **kwargs)

