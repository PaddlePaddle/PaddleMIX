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


import os
import re
from typing import Dict
from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import Qwen2VLImageProcessor, Qwen2VLProcessor, process_vision_info
from ...core import T, MMDataset, register
import paddle
from tqdm import tqdm



class QNAProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, dtype="bfloat16")
        self.image_processor = Qwen2VLImageProcessor()
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        self.processor = Qwen2VLProcessor(self.image_processor, self.tokenizer)

    def generate_qna_for_image(self, image_path: str) -> Dict:
        """
        Generate question-and-answer pairs for a single image.
        """
        # Prepare model inputs
        image_inputs, video_inputs = process_vision_info([
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ])

        # Instruction content
        instruction = """
        You are an AI visual assistant, and you are seeing a single image. Answer all questions as you are seeing the image.

        Please generate exactly 3 complete question-and-answer pairs about this image, following this structure:

        Q1: [First question about visual content]
        A1: [Detailed answer]

        Q2: [Second question about object relationships or counting]
        A2: [Detailed answer]

        Q3: [Third question about background knowledge or events]
        A3: [Detailed answer]

        For each question:
        1. Only ask about content that can be definitively answered from the image
        2. Provide detailed, multi-sentence answers
        3. Include specific visual details from the image
        4. Maintain an AI assistant perspective when answering

        Make sure each question type is different:
        - First question about basic visual content
        - Second question about object relationships or quantities
        - Third question about deeper context or implications
        """

        # Combine instruction and question
        image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{image_pad_token}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )

        with paddle.no_grad():
            # Inference to generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=1280)
            output_text = self.processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
        # Use regular expression to parse the output and extract Q&A pairs
        qna_pairs = []
        output_text = output_text[0]

        # Match Q&A pairs with regular expression
        qna_pattern = r"(Q\d:.*?)(A\d:.*?)(?=Q\d:|$)"
        matches = re.findall(qna_pattern, output_text, re.DOTALL)

        # Process each matched Q&A pair
        for i, match in enumerate(matches):
            question = match[0].strip().replace("Q1:", "").replace("Q2:", "").replace("Q3:", "").strip()
            answer = match[1].strip().replace("A1:", "").replace("A2:", "").replace("A3:", "").strip()

            # Skip empty question or answer pairs
            if not question or not answer:
                continue

            # Add <image>\n to the first question
            if i == 0:
                question = f"<image>\n{question}"

            qna_pairs.append([question, answer])

        # Return the image path and corresponding Q&A pairs
        return {
            "image": image_path,
            "conversations": qna_pairs
        }



@register()
def generate_qna_for_images(image_folder_path: str, model_name: str = "Qwen/Qwen2-VL-7B-Instruct") -> MMDataset:
    """
    Generate question-and-answer pairs for each image in the given folder path and return a dataset containing image paths and Q&A pairs.
    
    Parameters:
        image_folder_path (str): Folder path containing images to process.
        model_name (str): Model name to use.
        
    Returns:
        MMDataset: Generated dataset containing image paths and their corresponding Q&A pairs.
    """
    print("Generating Q&A pairs...")

    # Initialize QNAProcessor
    qna_processor = QNAProcessor(model_name=model_name)

    # Get all image files in the folder
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Generate Q&A pairs for each image
    qna_data = []
    for image_path in tqdm(image_paths):
        qna_pair = qna_processor.generate_qna_for_image(image_path)
        qna_data.append(qna_pair)
        print(qna_pair)

    # Combine image paths and Q&A pairs into dataset entries
    dataset = MMDataset(qna_data)

    return dataset
