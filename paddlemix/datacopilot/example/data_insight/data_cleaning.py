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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        dataset_dir: str = None,
        similarity_threshold: float = 0.95,
        image_quality_threshold: float = 0.7,
        text_quality_threshold: float = 0.6,
        matching_threshold: float = 0.3,
    ):
        """Initialize DataCleaner with configurable thresholds for data cleaning.

        Args:
            clip_model_name (str, optional): Name of the CLIP model to use for feature extraction.
                Defaults to "openai/clip-vit-base-patch32".
            dataset_dir (str, optional): Base directory containing dataset images. If None,
                image paths are treated as absolute paths. Defaults to None.
            similarity_threshold (float, optional): Threshold for duplicate detection (0-1).
                Samples with similarity scores above this will be considered duplicates.
                Defaults to 0.95.
            image_quality_threshold (float, optional): Minimum quality score (0-1) for images.
                Samples below this threshold will be filtered out. Defaults to 0.7.
            text_quality_threshold (float, optional): Minimum quality score (0-1) for text.
                Samples below this threshold will be filtered out. Defaults to 0.6.
            matching_threshold (float, optional): Minimum image-text matching score (0-1).
                Samples below this threshold will be filtered out. Defaults to 0.3.

        Note:
            All thresholds are inclusive - samples with scores equal to the threshold
            will be kept.
        """
        self.analyzer = DataAnalyzer(clip_model_name=clip_model_name, dataset_dir=dataset_dir)
        self.similarity_threshold = similarity_threshold
        self.image_quality_threshold = image_quality_threshold
        self.text_quality_threshold = text_quality_threshold
        self.matching_threshold = matching_threshold
        self.dataset_dir = dataset_dir

    def _get_image_path(self, image_name: str) -> str:
        """Construct full image path by combining dataset directory and image name.

        Args:
            image_name (str): Name or relative path of the image file

        Returns:
            str: Full path to the image file

        Note:
            If dataset_dir is None, returns image_name as-is (treating it as absolute path)
        """
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, image_name)
        return image_name

    def clean(self, data_path: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Clean dataset by removing duplicates and low quality samples.

        Performs the following cleaning steps in order:
        1. Anomaly detection - removes samples with invalid structure or missing files
        2. Duplicate detection - removes samples with similar image/text features
        3. Quality filtering - removes samples with low image or text quality
        4. Matching filtering - removes samples with low image-text matching scores

        Args:
            data_path (str): Path to the JSON file containing samples

        Returns:
            Tuple[List[Dict], Dict[str, int]]: Tuple containing:
                - List of cleaned samples that passed all filters
                - Dictionary of cleaning statistics including:
                    - initial_count: Total samples before cleaning
                    - removed_by_anomaly: Samples removed in anomaly detection
                    - removed_by_duplicate: Samples removed as duplicates
                    - removed_by_quality: Samples removed by quality filters
                    - removed_by_matching: Samples removed by matching filter
                    - final_count: Total samples after cleaning

        Raises:
            FileNotFoundError: If data_path does not exist
            json.JSONDecodeError: If JSON file is invalid
        """
        try:
            # Load data
            with open(data_path, "r") as f:
                samples = json.load(f)

            initial_count = len(samples)
            stats = {
                "initial_count": initial_count,
                "removed_by_anomaly": 0,
                "removed_by_duplicate": 0,
                "removed_by_quality": 0,
                "removed_by_matching": 0,
            }

            # Step 1: Handle anomalies
            valid_samples = self.handle_anomalies(samples)
            stats["removed_by_anomaly"] = initial_count - len(valid_samples)

            # Step 2: Remove duplicates
            unique_samples = self.detect_duplicates(valid_samples)
            stats["removed_by_duplicate"] = len(valid_samples) - len(unique_samples)

            # Step 3: Quality filtering
            quality_samples = self.filter_by_quality(unique_samples)
            stats["removed_by_quality"] = len(unique_samples) - len(quality_samples)

            # Step 4: Image-text matching
            final_samples = self.filter_by_matching(quality_samples)
            stats["removed_by_matching"] = len(quality_samples) - len(final_samples)

            stats["final_count"] = len(final_samples)
            logger.info(f"Cleaning complete. Stats: {stats}")
            return final_samples, stats

        except Exception as e:
            logger.error(f"Error cleaning data: {e}", exc_info=True)
            return [], {"error": str(e)}

    def handle_anomalies(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples with missing or invalid data.

        Args:
            samples (List[Dict]): List of samples to check

        Returns:
            List[Dict]: List of samples that passed anomaly checks

        Note:
            A sample is considered valid if it has:
            - Valid structure (id, image, conversations fields)
            - Existing image file
            - Valid conversation turns with 'from' and 'value' fields
        """
        cleaned_samples = []

        for sample in samples:
            try:
                # Validate sample structure
                if not self.analyzer._validate_sample(sample):
                    continue

                # Validate image path
                image_path = self._get_image_path(sample["image"])
                if not os.path.exists(image_path) or not os.path.isfile(image_path):
                    continue

                # Validate conversations
                if not sample["conversations"] or not all(
                    isinstance(conv, dict) and "from" in conv and "value" in conv for conv in sample["conversations"]
                ):
                    continue

                cleaned_samples.append(sample)

            except Exception as e:
                logger.warning(f"Error handling sample: {e}")

        return cleaned_samples

    def detect_duplicates(self, samples: List[Dict]) -> List[Dict]:
        """Detect and remove duplicate samples using image and text features.

        Args:
            samples (List[Dict]): List of samples to check for duplicates

        Returns:
            List[Dict]: List of unique samples

        Note:
            Duplicates are detected by comparing image and text feature vectors
            using cosine similarity. Samples with similarity scores above the
            configured threshold are considered duplicates.
        """
        try:
            unique_samples = []
            seen_features = []

            for sample in samples:
                image_path = self._get_image_path(sample["image"])
                text = " ".join(conv["value"] for conv in sample["conversations"])

                # Get image and text features
                matching_info = self.analyzer.analyze_image_text_matching(image_path, text)
                if not matching_info:
                    continue

                # Check for duplicates using feature similarity
                is_duplicate = False
                for prev_features in seen_features:
                    image_similarity = self._compute_similarity(
                        matching_info["image_features"], prev_features["image_features"]
                    )
                    text_similarity = self._compute_similarity(
                        matching_info["text_features"], prev_features["text_features"]
                    )

                    if image_similarity > self.similarity_threshold and text_similarity > self.similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    seen_features.append(matching_info)
                    unique_samples.append(sample)

            return unique_samples

        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}", exc_info=True)
            return samples

    def filter_by_quality(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on image and text quality metrics.

        Args:
            samples (List[Dict]): List of samples to filter

        Returns:
            List[Dict]: List of samples that meet quality thresholds

        Note:
            Both image and text quality must meet their respective thresholds.
            If any conversation turn fails the text quality check, the entire
            sample is filtered out.
        """
        filtered_samples = []

        for sample in samples:
            try:
                # Check image quality
                image_path = self._get_image_path(sample["image"])
                img_quality = self.analyzer.analyze_image_quality(image_path)
                if img_quality["quality_score"] < self.image_quality_threshold:
                    continue

                # Check text quality for each conversation
                keep_sample = True
                for conv in sample["conversations"]:
                    if "value" in conv:
                        text_quality = self.analyzer.analyze_text_quality(conv["value"])
                        if text_quality["text_score"] < self.text_quality_threshold:
                            keep_sample = False
                            break

                if keep_sample:
                    filtered_samples.append(sample)

            except Exception as e:
                logger.warning(f"Error in quality filtering for sample: {e}")

        return filtered_samples

    def filter_by_matching(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on image-text matching score.

        Args:
            samples (List[Dict]): List of samples to filter

        Returns:
            List[Dict]: List of samples with matching scores above threshold

        Note:
            The matching score is calculated using CLIP model embeddings
            for both the image and text content.
        """
        matched_samples = []

        for sample in samples:
            try:
                image_path = self._get_image_path(sample["image"])
                text = " ".join(conv["value"] for conv in sample["conversations"])

                matching_info = self.analyzer.analyze_image_text_matching(image_path, text)
                if matching_info and matching_info["similarity_score"] >= self.matching_threshold:
                    matched_samples.append(sample)

            except Exception as e:
                logger.warning(f"Error in matching filter for sample: {e}")

        return matched_samples

    def _compute_similarity(self, features1, features2) -> float:
        """Compute cosine similarity between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            float: Cosine similarity score between -1 and 1

        Note:
            Adds small epsilon (1e-8) to denominator for numerical stability
        """
        return float(
            (features1 * features2).sum() / (((features1**2).sum() ** 0.5) * ((features2**2).sum() ** 0.5) + 1e-8)
        )
