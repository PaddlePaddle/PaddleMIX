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
from typing import Dict, List

import numpy as np
from data_analysis import DataAnalyzer


class DataCleaner:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        self.analyzer = DataAnalyzer(clip_model_name)
        self.similarity_threshold = 0.95  # Threshold for duplicate detection
        self.quality_threshold = 0.7  # Threshold for quality filtering

    def detect_duplicates(self, samples: List[Dict]) -> List[int]:
        """Detect duplicate samples"""
        try:
            # Extract features
            features = []
            for sample in samples:
                result = self.analyzer.analyze_image_text_matching(sample["image_path"], sample["text"])
                if result and isinstance(result["image_features"], np.ndarray):
                    features.append(result["image_features"].flatten())

            if not features:
                return []

            # Calculate similarity matrix
            features = np.array(features)
            if len(features.shape) < 2:
                return []

            similarity_matrix = np.dot(features, features.T)

            # Mark duplicate samples
            duplicates = set()
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i, j] > self.similarity_threshold:
                        duplicates.add(j)

            return list(duplicates)
        except Exception as e:
            print(f"Error detecting duplicates: {e}")
            return []

    def filter_by_quality(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on quality score"""
        try:
            filtered_samples = []
            for sample in samples:
                # Evaluate image quality
                img_quality = self.analyzer.analyze_image_quality(sample["image_path"])
                if img_quality and img_quality["quality_score"] < self.quality_threshold:
                    continue

                # Evaluate text quality
                text_quality = self.analyzer.analyze_text_quality(sample["text"])
                if text_quality and text_quality["text_score"] < self.quality_threshold:
                    continue

                filtered_samples.append(sample)
            return filtered_samples
        except Exception as e:
            print(f"Error filtering by quality: {e}")
            return samples

    def handle_anomalies(self, samples: List[Dict]) -> List[Dict]:
        """Handle anomalous samples"""
        try:
            cleaned_samples = []
            for sample in samples:
                # Check if image path exists
                if not sample.get("image_path"):
                    continue

                # Check if text is empty
                if not sample.get("text") or len(sample["text"].strip()) == 0:
                    continue

                cleaned_samples.append(sample)
            return cleaned_samples
        except Exception as e:
            print(f"Error handling anomalies: {e}")
            return samples

    def clean(self, data_path: str) -> List[Dict]:
        """Clean dataset by removing duplicates and low quality samples"""
        try:
            # Load data
            with open(data_path, "r") as f:
                samples = json.load(f)

            # Handle basic anomalies first
            samples = self.handle_anomalies(samples)

            # Remove duplicates
            duplicate_indices = self.detect_duplicates(samples)
            samples = [s for i, s in enumerate(samples) if i not in duplicate_indices]

            # Filter by quality
            filtered_samples = []
            for sample in samples:
                # Check image quality
                if "image" in sample:
                    img_quality = self.analyzer.analyze_image_quality(sample["image"])
                    if img_quality["quality_score"] < self.quality_threshold:
                        continue

                # Check text quality
                valid_conversations = []
                for conv in sample["conversations"]:
                    if "value" in conv:
                        text_quality = self.analyzer.analyze_text_quality(conv["value"])
                        if text_quality and text_quality["text_score"] >= self.quality_threshold:
                            valid_conversations.append(conv)

                if valid_conversations:
                    sample["conversations"] = valid_conversations
                    filtered_samples.append(sample)

            return filtered_samples

        except Exception as e:
            print(f"Error cleaning data: {e}")
            return []
