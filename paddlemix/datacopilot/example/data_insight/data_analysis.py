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

import json
import os
from typing import Dict, List, Union

import cv2
import numpy as np
import paddle
from paddlenlp.transformers import AutoTokenizer
from PIL import Image

from paddlemix.models.clip.clip_model import CLIP


class DataAnalyzer:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        self.clip_model = CLIP.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)

    def analyze_image_quality(self, image_path):
        """Evaluate image quality"""
        try:
            # Open image file directly using PIL
            img = Image.open(image_path)

            # Convert to grayscale for Laplacian analysis
            img_gray = np.array(img.convert("L"))
            laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

            # Get resolution
            width, height = img.size
            resolution_score = min(width, height) / 1024

            # Calculate quality score
            quality_score = 0.6 * laplacian_var + 0.4 * resolution_score

            return {"quality_score": quality_score, "laplacian_var": laplacian_var, "resolution": (width, height)}
        except Exception as e:
            print(f"Error analyzing image quality for {image_path}: {str(e)}")
            return {"quality_score": 0.0, "laplacian_var": 0.0, "resolution": (0, 0)}

    def analyze_text_quality(self, text):
        """Evaluate text quality"""
        try:
            # Basic text quality metrics instead of using CLIP
            text_length = len(text.split())
            has_punctuation = any(p in text for p in ".!?")
            complexity = min(text_length / 50, 1.0)

            text_score = 0.4 * complexity + 0.3 * (text_length > 5) + 0.3 * has_punctuation

            return {"text_score": text_score, "complexity": complexity, "length": text_length}
        except Exception as e:
            print(f"Error analyzing text quality: {e}")
            return None

    def analyze_image_text_matching(self, image_path, text):
        """Evaluate image-text matching"""
        try:
            # Preprocess image
            img = Image.open(image_path)
            image_inputs = self.clip_model.processor(images=img, return_tensors="pd")

            # Preprocess text
            text_inputs = self.tokenizer(text, return_tensors="pd")

            # Calculate similarity
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)

            # Calculate cosine similarity
            similarity = paddle.nn.functional.cosine_similarity(image_features, text_features).item()

            return {
                "similarity_score": similarity,
                "image_features": image_features.numpy(),
                "text_features": text_features.numpy(),
            }
        except Exception as e:
            print(f"Error analyzing image-text matching: {e}")
            return None

    def detect_language(self, text: str) -> str:
        """Detect language of text using simple heuristics"""
        try:
            # Check for common language patterns
            if any(ord(c) > 0x4E00 for c in text):
                return "zh"  # Chinese
            elif any(0x0600 <= ord(c) <= 0x06FF for c in text):
                return "ar"  # Arabic
            elif any(0x0900 <= ord(c) <= 0x097F for c in text):
                return "hi"  # Hindi
            elif all(ord(c) < 128 for c in text):
                return "en"  # English
            else:
                return "other"
        except:
            return "unknown"

    def analyze(self, data: Union[str, List[Dict]]) -> Dict:
        """Analyze dataset statistics from JSON file or list"""
        try:
            if isinstance(data, str):
                with open(data, "r") as f:
                    samples = json.load(f)
            else:
                samples = data

            stats = {
                "total_samples": len(samples),
                "valid_samples": 0,
                "invalid_samples": 0,
                "avg_conversations": 0,
                "avg_text_length": 0,
                "language_dist": {},
                "format_dist": {},
            }

            valid_conv_lengths = []
            valid_text_lengths = []

            for sample in samples:
                # Check required fields
                if not all(k in sample for k in ["id", "image", "conversations"]):
                    stats["invalid_samples"] += 1
                    continue

                # Check conversations format
                conversations = sample["conversations"]
                if not isinstance(conversations, list):
                    stats["invalid_samples"] += 1
                    continue

                # Analyze conversations
                valid_conv_lengths.append(len(conversations))

                # Analyze text lengths and languages
                for conv in conversations:
                    if "value" in conv:
                        text = conv["value"]
                        valid_text_lengths.append(len(text))

                        # Detect language
                        lang = self.detect_language(text)
                        stats["language_dist"][lang] = stats["language_dist"].get(lang, 0) + 1

                # Analyze image format
                if "image" in sample:
                    img_ext = os.path.splitext(sample["image"])[1].lower()
                    stats["format_dist"][img_ext] = stats["format_dist"].get(img_ext, 0) + 1

                stats["valid_samples"] += 1

            # Calculate averages
            if valid_conv_lengths:
                stats["avg_conversations"] = sum(valid_conv_lengths) / len(valid_conv_lengths)
            if valid_text_lengths:
                stats["avg_text_length"] = sum(valid_text_lengths) / len(valid_text_lengths)

            return stats

        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "invalid_samples": 0,
                "avg_conversations": 0,
                "avg_text_length": 0,
                "language_dist": {},
                "format_dist": {},
            }
