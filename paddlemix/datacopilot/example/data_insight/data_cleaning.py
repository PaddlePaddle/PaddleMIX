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
import logging
import os
from typing import Dict, List, Tuple

from data_analysis import DataAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        similarity_threshold: float = 0.95,
        quality_threshold: float = 0.7,
        dataset_dir: str = None,
    ):
        """Initialize DataCleaner with configurable thresholds.

        Args:
            clip_model_name: Name of the CLIP model to use
            similarity_threshold: Threshold for duplicate detection (0-1)
            quality_threshold: Threshold for quality filtering (0-1)
            dataset_dir: Base directory containing the dataset images
        """
        self.analyzer = DataAnalyzer(clip_model_name, dataset_dir)
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold
        self.dataset_dir = dataset_dir

    def _get_image_path(self, image_name: str) -> str:
        """Get full image path by combining dataset directory and image name."""
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, image_name)
        return image_name

    def clean(self, data_path: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Clean dataset by removing duplicates and low quality samples.

        Args:
            data_path: Path to the JSON file containing samples

        Returns:
            Tuple of (cleaned_samples, stats)
            - cleaned_samples: List of samples that passed all cleaning steps
            - stats: Dictionary containing cleaning statistics
        """
        try:
            # Load data
            with open(data_path, "r") as f:
                samples = json.load(f)

            # Update image paths
            for sample in samples:
                if "image" in sample:
                    sample["image"] = self._get_image_path(sample["image"])

            initial_count = len(samples)
            stats = {"initial_count": initial_count}

            # Step 1: Handle anomalies
            samples = self.handle_anomalies(samples)
            stats["after_anomaly_count"] = len(samples)

            # Step 2: Remove duplicates
            duplicate_indices = self.detect_duplicates(samples)
            samples = [s for i, s in enumerate(samples) if i not in duplicate_indices]
            stats["after_dedup_count"] = len(samples)

            # Step 3: Quality filtering
            samples = self.filter_by_quality(samples)
            stats["final_count"] = len(samples)

            logger.info(f"Cleaning complete. Stats: {stats}")
            return samples, stats

        except Exception as e:
            logger.error(f"Error cleaning data: {e}", exc_info=True)
            return [], {"error": str(e)}

    def handle_anomalies(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples with missing or invalid data."""
        cleaned_samples = []
        anomaly_count = 0

        for sample in samples:
            try:
                # Validate sample structure
                if not isinstance(sample, dict):
                    anomaly_count += 1
                    continue

                # Check required fields
                if not all(k in sample for k in ["image_path", "text"]):
                    anomaly_count += 1
                    continue

                # Validate image path
                if not os.path.exists(sample["image_path"]):
                    anomaly_count += 1
                    continue

                # Validate text
                if not isinstance(sample["text"], str) or not sample["text"].strip():
                    anomaly_count += 1
                    continue

                cleaned_samples.append(sample)

            except Exception as e:
                logger.warning(f"Error handling sample: {e}")
                anomaly_count += 1

        logger.info(f"Removed {anomaly_count} anomalous samples")
        return cleaned_samples

    def detect_duplicates(self, samples: List[Dict]) -> List[Dict]:
        """Detect and remove duplicate samples"""
        try:
            unique_samples = []
            seen_hashes = set()

            for sample in samples:
                # Generate hash based on image and text content
                if "image" in sample and "text" in sample:
                    content_hash = hash(str(sample["image"]) + sample["text"])

                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        unique_samples.append(sample)
                else:
                    unique_samples.append(sample)  # Keep samples without image/text pairs

            return unique_samples
        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}", exc_info=True)
            return samples

    def filter_by_quality(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on quality metrics"""
        try:
            filtered_samples = []
            for sample in samples:
                keep_sample = True

                # Check image quality if image exists
                if "image" in sample:
                    img_quality = self.analyzer.analyze_image_quality(sample["image"])
                    if img_quality and img_quality.get("quality_score", 0) < self.quality_threshold:
                        keep_sample = False

                # Check text quality if text exists
                if "text" in sample and keep_sample:
                    text_quality = self.analyzer.analyze_text_quality(sample["text"])
                    if text_quality and text_quality.get("text_score", 0) < self.quality_threshold:
                        keep_sample = False

                if keep_sample:
                    filtered_samples.append(sample)

            return filtered_samples
        except Exception as e:
            logger.error(f"Error filtering by quality: {e}", exc_info=True)
            return samples
