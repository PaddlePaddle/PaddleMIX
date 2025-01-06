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

import logging
import os
import warnings
from typing import Any, Dict, List, Union

import ijson  # For streaming JSON parsing
from paddlenlp.transformers import AutoTokenizer, CLIPProcessor
from PIL import Image
from tqdm import tqdm

from paddlemix.models.clip.clip_model import CLIP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class DataAnalyzer:
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", dataset_dir: str = None):
        """Initialize DataAnalyzer with CLIP model for image-text analysis.

        Args:
            clip_model_name: Name or path of the CLIP model to use
            dataset_dir: Base directory containing the dataset images
        """
        try:
            self.clip_model = CLIP.from_pretrained(clip_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.dataset_dir = dataset_dir
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _get_image_path(self, image_name: str) -> str:
        """Get full image path by combining dataset directory and image name."""
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, image_name)
        return image_name

    def analyze(self, data: Union[str, List[Dict]], chunk_size: int = 1000) -> Dict:
        """Analyze dataset statistics from JSON file or list."""
        try:
            # Initialize statistics
            stats = self._get_empty_stats()

            # Stream process JSON file or iterate through list
            if isinstance(data, str):
                with open(data, "rb") as file:
                    samples = ijson.items(file, "item")
                    self._process_samples(samples, stats, show_progress=True)
            else:
                self._process_samples(data, stats, show_progress=True)

            # Calculate final averages and clean up statistics
            self._finalize_statistics(stats)

            return stats
        except Exception as e:
            warnings.warn(f"Error analyzing dataset: {e}")
            return self._get_empty_stats()

    def _process_samples(
        self, samples: Union[List[Dict], Any], stats: Dict[str, Any], show_progress: bool = False
    ) -> None:
        """Process dataset samples and update statistics."""
        iterator = tqdm(samples) if show_progress else samples

        for sample in iterator:
            stats["total_samples"] += 1
            sample_id = sample.get("id", "unknown")

            try:
                if not self._validate_sample(sample):
                    stats["invalid_samples"] += 1
                    logger.warning(f"Invalid sample structure: {sample_id}")
                    continue

                # Get full image path
                if "image" in sample:
                    sample["image"] = self._get_image_path(sample["image"])

                stats["valid_samples"] += 1
                self._analyze_conversations(sample, stats)
                self._analyze_image_format(sample, stats)

            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {str(e)}")
                stats["errors"][str(e)] += 1
                stats["invalid_samples"] += 1
            finally:
                continue

    def _validate_sample(self, sample: Dict) -> bool:
        """Validate sample structure"""
        required_fields = ["id", "image", "conversations"]
        return all(k in sample for k in required_fields) and isinstance(sample["conversations"], list)

    def _analyze_conversations(self, sample: Dict, stats: Dict):
        """Analyze conversation metrics"""
        conversations = sample["conversations"]
        turn_count = len(conversations)

        # Update conversation statistics
        stats["conversation_stats"]["min_turns"] = min(stats["conversation_stats"]["min_turns"], turn_count)
        stats["conversation_stats"]["max_turns"] = max(stats["conversation_stats"]["max_turns"], turn_count)
        stats["conversation_stats"]["total_turns"] += turn_count
        stats["conversation_stats"]["turn_distribution"][turn_count] += 1

        # Analyze each conversation turn
        for conv in conversations:
            if "value" in conv:
                text = conv["value"]
                text_length = len(text)

                # Update text statistics
                stats["text_stats"]["min_length"] = min(stats["text_stats"]["min_length"], text_length)
                stats["text_stats"]["max_length"] = max(stats["text_stats"]["max_length"], text_length)
                stats["text_stats"]["total_length"] += text_length

                # Update language distribution
                lang_info = self.detect_language(text)
                for lang, prop in lang_info["lang_proportions"].items():
                    stats["language_dist"][lang] += prop

    def _analyze_image_format(self, sample: Dict, stats: Dict):
        """Analyze image format statistics"""
        if "image" in sample:
            img_ext = os.path.splitext(sample["image"])[1].lower()
            stats["format_dist"][img_ext] += 1

    def _finalize_statistics(self, stats: Dict):
        """Calculate final averages and clean up statistics"""
        valid_samples = max(1, stats["valid_samples"])  # Avoid division by zero

        # Calculate averages
        stats["avg_conversations"] = stats["conversation_stats"]["total_turns"] / valid_samples
        stats["avg_text_length"] = stats["text_stats"]["total_length"] / valid_samples

        # Convert defaultdicts to regular dicts
        stats["language_dist"] = dict(stats["language_dist"])
        stats["format_dist"] = dict(stats["format_dist"])
        stats["conversation_stats"]["turn_distribution"] = dict(stats["conversation_stats"]["turn_distribution"])

        # Remove infinity values if no valid samples were processed
        if stats["conversation_stats"]["min_turns"] == float("inf"):
            stats["conversation_stats"]["min_turns"] = 0
        if stats["text_stats"]["min_length"] == float("inf"):
            stats["text_stats"]["min_length"] = 0

    def _get_empty_stats(self) -> Dict:
        """Return empty statistics structure"""
        return {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "conversation_stats": {"min_turns": 0, "max_turns": 0, "total_turns": 0, "turn_distribution": {}},
            "text_stats": {"min_length": 0, "max_length": 0, "total_length": 0},
            "avg_conversations": 0,
            "avg_text_length": 0,
            "language_dist": {},
            "format_dist": {},
            "errors": {},
        }

    def analyze_image_quality(self, image) -> Dict:
        """Analyze image quality metrics"""
        try:
            # Basic image quality metrics
            quality_metrics = {
                "resolution": image.size if hasattr(image, "size") else None,
                "aspect_ratio": image.size[0] / image.size[1] if hasattr(image, "size") else None,
                "quality_score": 1.0,  # Placeholder score, could implement more sophisticated metrics
            }
            return quality_metrics
        except Exception as e:
            print(f"Error analyzing image quality: {e}")
            return None

    def analyze_image_text_matching(self, image, text) -> Dict:
        """Analyze image-text matching score"""
        try:
            # Use CLIP model to compute similarity
            if hasattr(self, "model"):
                image_features = self.model.get_image_features(image)
                text_features = self.model.get_text_features(text)
                similarity = (image_features @ text_features.T).item()

                return {"matching_score": similarity, "is_matched": similarity > 0.5}  # Threshold can be adjusted
            return {"matching_score": 0.5, "is_matched": True}  # Default fallback score
        except Exception as e:
            print(f"Error analyzing image-text matching: {e}")
            return None

    def analyze_text_quality(self, text: str) -> Dict:
        """Analyze text quality metrics"""
        try:
            # Basic text quality metrics
            quality_metrics = {
                "length": len(text),
                "word_count": len(text.split()),
                "text_score": 1.0,  # Placeholder score, could implement more sophisticated metrics
            }
            return quality_metrics
        except Exception as e:
            print(f"Error analyzing text quality: {e}")
            return None

    def analyze_sample(self, sample: Dict) -> Dict:
        """Analyze a single sample"""
        try:
            # Validate sample format
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample format: {type(sample)}")

            if "image_path" not in sample or "text" not in sample:
                raise ValueError("Sample missing required fields")

            # Get image path and load image
            image_path = sample["image_path"]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path)

            # Basic analysis
            analysis = {
                "image_analysis": self.analyze_image(image),
                "text_analysis": self.analyze_text(sample["text"]),
                "image_text_matching": None,
            }

            # Optional CLIP-based analysis if model available
            if hasattr(self, "model"):
                analysis["image_text_matching"] = self.analyze_image_text_matching(image, sample["text"])

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing sample {sample.get('image_path', 'unknown')}: {str(e)}")
            return {"image_analysis": None, "text_analysis": None, "image_text_matching": None, "error": str(e)}

    def analyze_dataset(self, samples: List[Dict]) -> Dict:
        """Analyze full dataset"""
        try:
            stats = self._get_empty_stats()

            for sample in tqdm(samples, desc="Analyzing samples"):
                try:
                    analysis = self.analyze_sample(sample)

                    if all(v is not None for v in analysis.values()):
                        stats["valid_samples"] += 1
                        # Update statistics based on analysis...
                    else:
                        stats["invalid_samples"] += 1
                        if "error" in analysis:
                            error_type = type(analysis["error"]).__name__
                            stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1

                except Exception as e:
                    stats["invalid_samples"] += 1
                    error_type = type(e).__name__
                    stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
                    logger.warning(f"Error analyzing sample: {e}")

            stats["total_samples"] = len(samples)
            self._finalize_statistics(stats)
            return stats

        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            return self._get_empty_stats()
