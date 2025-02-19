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
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import cv2
import ijson  # For streaming JSON parsing
import langid
import numpy as np
import paddle
from paddlenlp.transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm

from ppdiffusers.transformers import CLIPImageProcessor  # Update import
from ppdiffusers.transformers import CLIPModel as CLIP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class DataAnalyzer:
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", dataset_dir: str = None):
        """Initialize DataAnalyzer for analyzing image-text pairs using CLIP model.

        Args:
            clip_model_name (str, optional): Name or path of the CLIP model to use. Defaults to "openai/clip-vit-base-patch32".
            dataset_dir (str, optional): Base directory containing dataset images. Defaults to None.

        Raises:
            RuntimeError: If CLIP model initialization fails
        """
        try:
            self.clip_model = CLIP.from_pretrained(clip_model_name)
            self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)  # Use new processor
            self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
            self.dataset_dir = dataset_dir
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def analyze_image_quality(self, image_path: str) -> Dict[str, Union[float, Tuple[int, int]]]:
        """Evaluate image quality using multiple metrics including blur, resolution, lighting, and color.

        Args:
            image_path (str): Path to the image file

        Returns:
            Dict[str, Union[float, Tuple[int, int]]]: Dictionary containing:
                - quality_score (float): Overall quality score between 0 and 1
                - blur_score (float): Blur detection score
                - resolution_score (float): Resolution quality score
                - brightness_score (float): Lighting quality score
                - contrast_score (float): Image contrast score
                - resolution (Tuple[int, int]): Image dimensions (width, height)
                - aspect_ratio (float): Image aspect ratio
                - noise_score (float): Image noise level score
                - color_score (float): Color quality score
                - saturation_score (float): Color saturation score
                - compression_score (float): Compression quality score
                - color_distribution (Dict): Color distribution analysis
        """
        try:
            img = Image.open(image_path)
            img_np = np.array(img)

            # Resize large images for faster processing
            img_np = self._preprocess_image(img_np)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np

            # Calculate individual metrics
            blur_metrics = self._calculate_blur_metrics(img_gray)
            resolution_metrics = self._calculate_resolution_metrics(img)
            lighting_metrics = self._calculate_lighting_metrics(img_gray)
            color_metrics = self._calculate_color_metrics(img_np)

            # Calculate final quality score
            quality_score = self._compute_overall_quality(
                blur_metrics, resolution_metrics, lighting_metrics, color_metrics
            )

            return {
                **blur_metrics,
                **resolution_metrics,
                **lighting_metrics,
                **color_metrics,
                "quality_score": quality_score,
            }
        except Exception as e:
            logger.error(f"Image quality analysis failed for {image_path}: {str(e)}")
            return self._get_empty_image_metrics()

    def _preprocess_image(self, img_np: np.ndarray) -> np.ndarray:
        """Resize image if needed for efficient processing."""
        max_dimension = 1024
        h, w = img_np.shape[:2]
        scale = max_dimension / max(h, w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(img_np, (new_w, new_h))
        return img_np

    def _calculate_blur_metrics(self, img_gray: np.ndarray) -> Dict[str, float]:
        """Calculate blur-related metrics."""
        img_filtered = cv2.bilateralFilter(img_gray, 9, 75, 75)
        laplacian_var = cv2.Laplacian(img_filtered, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500)
        return {"blur_score": float(blur_score)}

    def _calculate_resolution_metrics(self, img: Image.Image) -> Dict[str, Union[float, Tuple[int, int]]]:
        """Calculate resolution-related metrics."""
        width, height = img.size
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        aspect_ratio = min_dimension / max_dimension
        resolution_score = min(1.0, (min_dimension / 1024) * (1 + aspect_ratio) / 2)
        return {
            "resolution_score": float(resolution_score),
            "resolution": (width, height),
            "aspect_ratio": float(aspect_ratio),
        }

    def _calculate_lighting_metrics(self, img_gray: np.ndarray) -> Dict[str, float]:
        """Calculate lighting-related metrics."""
        brightness = np.mean(img_gray)
        contrast = np.std(img_gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        contrast_score = min(1.0, contrast / 128)
        return {"brightness_score": float(brightness_score), "contrast_score": float(contrast_score)}

    def _calculate_color_metrics(self, img_np: np.ndarray) -> Dict[str, Union[float, Dict]]:
        """Calculate color-related metrics."""
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        noise_score = self._calculate_noise_score(img_np)
        color_distribution = self._analyze_color_distribution(img_np)
        saturation = img_hsv[:, :, 1].mean() / 255
        compression_score = self._estimate_compression_quality(img_np)

        color_score = (
            0.4 * color_distribution["distribution_score"] + 0.3 * saturation + 0.3 * (1 - abs(saturation - 0.5))
        )

        return {
            "noise_score": float(noise_score),
            "color_score": float(color_score),
            "saturation_score": float(saturation),
            "compression_score": float(compression_score),
            "color_distribution": color_distribution,
        }

    def _calculate_noise_score(self, img_np: np.ndarray) -> float:
        """Calculate image noise score."""
        noise_sigma = self._estimate_noise(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY))
        return max(0, min(1, 1 - noise_sigma / 30))

    def _estimate_noise(self, img_gray: np.ndarray) -> float:
        """Estimate image noise level using median absolute deviation."""
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

        # Calculate median absolute deviation (MAD)
        median = np.median(np.abs(laplacian))
        mad = np.median(np.abs(laplacian - median))

        # Convert MAD to noise estimate (sigma)
        noise_sigma = mad * 1.4826  # Constant for normal distribution

        return float(noise_sigma)

    def _analyze_color_distribution(self, img_np: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution of the image."""
        try:
            # Convert to HSV for better color analysis
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

            # Calculate color histograms
            h_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])

            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
            s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
            v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)

            # Calculate distribution scores
            color_variety = float(np.count_nonzero(h_hist > 0.05) / 180)
            saturation_spread = float(np.std(s_hist))
            value_spread = float(np.std(v_hist))

            # Calculate overall distribution score
            distribution_score = (color_variety + saturation_spread + value_spread) / 3

            return {
                "distribution_score": float(distribution_score),
                "color_variety": color_variety,
                "saturation_spread": saturation_spread,
                "value_spread": value_spread,
                "hue_histogram": h_hist.flatten().tolist(),
                "saturation_histogram": s_hist.flatten().tolist(),
                "value_histogram": v_hist.flatten().tolist(),
            }
        except Exception as e:
            logger.warning(f"Error analyzing color distribution: {e}")
            return {
                "distribution_score": 0.0,
                "color_variety": 0.0,
                "saturation_spread": 0.0,
                "value_spread": 0.0,
                "hue_histogram": [],
                "saturation_histogram": [],
                "value_histogram": [],
            }

    def _estimate_compression_quality(self, img_np: np.ndarray) -> float:
        """Estimate image compression quality."""
        try:
            # Convert to grayscale for DCT analysis
            if len(img_np.shape) == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np

            # Apply DCT transform
            dct = cv2.dct(np.float32(img_gray))

            # Calculate high-frequency component ratio
            total_energy = np.sum(np.abs(dct))
            high_freq_energy = np.sum(np.abs(dct[32:, 32:]))

            # Compute quality score (higher ratio of high frequencies indicates less compression)
            quality_score = 1.0 - (high_freq_energy / (total_energy + 1e-8))

            return float(quality_score)
        except Exception as e:
            logger.warning(f"Error estimating compression quality: {e}")
            return 0.0

    def _compute_overall_quality(
        self,
        blur_metrics: Dict[str, float],
        resolution_metrics: Dict[str, Union[float, Tuple[int, int]]],
        lighting_metrics: Dict[str, float],
        color_metrics: Dict[str, Union[float, Dict]],
    ) -> float:
        """Compute overall image quality score."""
        weights = {"blur": 0.3, "resolution": 0.2, "brightness": 0.15, "contrast": 0.15, "noise": 0.1, "color": 0.1}

        return float(
            weights["blur"] * blur_metrics["blur_score"]
            + weights["resolution"] * resolution_metrics["resolution_score"]
            + weights["brightness"] * lighting_metrics["brightness_score"]
            + weights["contrast"] * lighting_metrics["contrast_score"]
            + weights["noise"] * color_metrics["noise_score"]
            + weights["color"] * color_metrics["color_score"]
        )

    def _get_empty_image_metrics(self) -> Dict[str, Union[float, Tuple[int, int], Dict]]:
        """Return empty image metrics structure."""
        return {
            "quality_score": 0.0,
            "blur_score": 0.0,
            "resolution_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "resolution": (0, 0),
            "aspect_ratio": 0.0,
            "noise_score": 0.0,
            "color_score": 0.0,
            "saturation_score": 0.0,
            "compression_score": 0.0,
            "color_distribution": {},
        }

    def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Evaluate text quality using linguistic and readability metrics.

        Args:
            text (str): Input text to analyze

        Returns:
            Dict[str, Any]: Dictionary containing:
                - text_score (float): Overall text quality score between 0 and 1
                - metrics (Dict): Detailed metrics including:
                    - length (int): Word count
                    - char_length (int): Character count
                    - avg_sentence_length (float): Average words per sentence
                    - vocabulary_diversity (float): Unique words ratio
                    - punctuation_ratio (float): Punctuation density
                    - readability_score (float): Text readability score
        """
        try:
            # Basic text metrics
            text_metrics = self._calculate_basic_text_metrics(text)
            sentence_metrics = self._calculate_sentence_metrics(text)
            vocabulary_metrics = self._calculate_vocabulary_metrics(text)
            style_metrics = self._calculate_style_metrics(text)

            # Calculate final quality score
            quality_score = self._compute_text_quality_score(
                text_metrics, sentence_metrics, vocabulary_metrics, style_metrics
            )

            return {
                "text_score": quality_score,
                "metrics": {**text_metrics, **sentence_metrics, **vocabulary_metrics, **style_metrics},
            }
        except Exception as e:
            logger.error(f"Text quality analysis failed: {str(e)}")
            return self._get_empty_text_metrics()

    def _calculate_basic_text_metrics(self, text: str) -> Dict[str, Union[int, float]]:
        """Calculate basic text metrics."""
        words = text.split()
        return {"length": len(words), "char_length": len(text)}

    def _calculate_sentence_metrics(self, text: str) -> Dict[str, float]:
        """Calculate sentence-related metrics."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        words = text.split()
        avg_sentence_length = len(words) / max(len(sentences), 1)
        return {"avg_sentence_length": float(avg_sentence_length)}

    def _calculate_vocabulary_metrics(self, text: str) -> Dict[str, float]:
        """Calculate vocabulary-related metrics."""
        words = text.split()
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / max(len(words), 1)
        return {"vocabulary_diversity": float(vocabulary_diversity)}

    def _calculate_style_metrics(self, text: str) -> Dict[str, float]:
        """Calculate style-related metrics."""
        punctuation_marks = ".!?;:,\"'"
        punctuation_count = sum(text.count(p) for p in punctuation_marks)
        punctuation_ratio = punctuation_count / max(len(text), 1)

        words = text.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        readability_score = max(0, min(1, 1 - (avg_word_length - 4) / 4))

        return {"punctuation_ratio": float(punctuation_ratio), "readability_score": float(readability_score)}

    def _compute_text_quality_score(
        self,
        text_metrics: Dict[str, Union[int, float]],
        sentence_metrics: Dict[str, float],
        vocabulary_metrics: Dict[str, float],
        style_metrics: Dict[str, float],
    ) -> float:
        """Compute overall text quality score."""
        complexity_factors = {
            "length": min(text_metrics["length"] / 50, 1.0),
            "sentence_length": min(1, 2 / (1 + np.exp(sentence_metrics["avg_sentence_length"] / 20))),
            "vocabulary": min(vocabulary_metrics["vocabulary_diversity"] * 2, 1.0),
            "punctuation": min(style_metrics["punctuation_ratio"] * 5, 1.0),
            "readability": style_metrics["readability_score"],
        }

        weights = {"length": 0.2, "sentence_length": 0.2, "vocabulary": 0.3, "punctuation": 0.15, "readability": 0.15}

        return float(sum(score * weights[metric] for metric, score in complexity_factors.items()))

    def _get_empty_text_metrics(self) -> Dict[str, Any]:
        """Return empty text metrics structure."""
        return {
            "text_score": 0.0,
            "metrics": {
                "length": 0,
                "char_length": 0,
                "avg_sentence_length": 0.0,
                "vocabulary_diversity": 0.0,
                "punctuation_ratio": 0.0,
                "readability_score": 0.0,
            },
        }

    def analyze_image_text_matching(self, image_path: str, text: str) -> Dict[str, Any]:
        """Evaluate semantic similarity between image and text using CLIP model.

        Args:
            image_path (str): Path to the image file
            text (str): Text to compare with the image

        Returns:
            Dict[str, Any]: Dictionary containing:
                - similarity_score (float): Cosine similarity score between image and text
                - image_features (np.ndarray): Extracted image feature vector
                - text_features (np.ndarray): Extracted text feature vector
                or None if analysis fails
        """
        try:
            # Preprocess image
            img = Image.open(image_path).convert("RGB")
            # Use CLIPImageProcessor for image processing
            image_inputs = self.processor(images=img, return_tensors="pd")

            # Preprocess text
            text_inputs = self.tokenizer([text], return_tensors="pd", padding=True, truncation=True)

            # Extract features
            with paddle.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                text_features = self.clip_model.get_text_features(**text_inputs)

            # Normalize features
            image_features = image_features / paddle.norm(image_features, axis=-1, keepdim=True)
            text_features = text_features / paddle.norm(text_features, axis=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = paddle.matmul(image_features, text_features.t()).item()

            return {
                "similarity_score": float(similarity),
                "image_features": image_features.numpy(),
                "text_features": text_features.numpy(),
            }
        except Exception as e:
            logger.error(f"Error analyzing image-text matching: {e}")
            return None

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of input text using langid library.

        Args:
            text (str): Input text for language detection

        Returns:
            Dict[str, Any]: Dictionary containing:
                - primary_lang (str): ISO code of primary detected language
                - confidence (float): Detection confidence score between 0 and 1
                - lang_proportions (Dict[str, float]): Language distribution
                - method (str): Detection method used
        """
        try:
            # Use langid for fast and accurate language detection
            lang, confidence = langid.classify(text)

            # Normalize confidence score to 0-1 range
            normalized_confidence = min(1.0, max(0.0, float(confidence) / 100.0))

            return {
                "primary_lang": lang,
                "confidence": normalized_confidence,
                "lang_proportions": {lang: 1.0},
                "method": "langid",
            }

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return {"primary_lang": "unknown", "confidence": 0.0, "lang_proportions": {}, "method": "error"}

    def analyze(self, data: Union[str, List[Dict]], chunk_size: int = 1000) -> Dict:
        """Analyze dataset statistics from JSON file or list of samples.

        Processes image-text pairs to gather comprehensive statistics about the dataset,
        including conversation patterns, language distribution, and file formats.

        Args:
            data (Union[str, List[Dict]]): JSON file path or list of data samples
            chunk_size (int, optional): Number of samples to process in each batch. Defaults to 1000.

        Returns:
            Dict: Statistics including:
                - total_samples (int): Total number of samples processed
                - valid_samples (int): Number of valid samples
                - invalid_samples (int): Number of invalid samples
                - image_stats (Dict): Image-related statistics
                - conversation_stats (Dict): Conversation turn statistics
                - text_stats (Dict): Text length and content statistics
                - language_dist (Dict): Language distribution
                - format_dist (Dict): File format distribution
                - errors (Dict): Error counts by type
        """
        try:
            # Initialize statistics with defaultdict for easier counting
            stats = {
                "total_samples": 0,
                "valid_samples": 0,
                "invalid_samples": 0,
                "image_stats": {
                    "missing_files": 0,
                    "invalid_files": 0,
                },
                "conversation_stats": {
                    "min_turns": float("inf"),
                    "max_turns": 0,
                    "total_turns": 0,
                    "turn_distribution": defaultdict(int),
                },
                "text_stats": {"min_length": float("inf"), "max_length": 0, "total_length": 0},
                "language_dist": defaultdict(float),
                "format_dist": defaultdict(int),
                "errors": defaultdict(int),
            }

            # Stream process JSON file or iterate through list
            if isinstance(data, str):
                # Use ijson for memory-efficient JSON parsing
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
        """Process dataset samples and update statistics.

        Args:
            samples: Iterator of dataset samples
            stats: Statistics dictionary to update
            show_progress: Whether to show progress bar
        """
        iterator = tqdm(samples) if show_progress else samples

        for sample in iterator:
            stats["total_samples"] += 1
            sample_id = sample.get("id", "unknown")

            try:
                if not self._validate_sample(sample):
                    stats["invalid_samples"] += 1
                    logger.warning(f"Invalid sample structure: {sample_id}")
                    continue

                stats["valid_samples"] += 1
                self._analyze_conversations(sample, stats)
                self._analyze_image_format(sample, stats)

                if "image" in sample and self.dataset_dir is not None:
                    image_path = os.path.join(self.dataset_dir, sample["image"])
                    if not os.path.exists(image_path):
                        stats["image_stats"]["missing_files"] += 1
                        logger.warning(f"Image file not found: {image_path}")
                    elif not os.path.isfile(image_path):
                        stats["image_stats"]["invalid_files"] += 1
                        logger.warning(f"Invalid image file: {image_path}")

            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {str(e)}", exc_info=True)  # Include stack trace
                stats["errors"][str(e)] += 1
                stats["invalid_samples"] += 1

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
