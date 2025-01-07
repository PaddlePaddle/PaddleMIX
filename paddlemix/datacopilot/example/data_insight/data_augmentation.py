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
import random
from typing import Dict, List

from paddlenlp.dataaug import WordDelete, WordSubstitute
from PIL import Image, ImageEnhance


class DataAugmentor:
    def __init__(self, dataset_dir: str = None):
        """Initialize DataAugmentor with configuration for image and text augmentation.

        Args:
            dataset_dir (str, optional): Base directory containing the dataset images.
                If None, image paths are treated as absolute paths. Defaults to None.
        """
        self.image_aug_prob = 0.5
        self.text_aug_prob = 0.3
        self.dataset_dir = dataset_dir

    def _get_image_path(self, image_name: str) -> str:
        """Construct full image path by combining dataset directory and image name.

        Args:
            image_name (str): Name or relative path of the image file

        Returns:
            str: Full path to the image file
        """
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, image_name)
        return image_name

    def augment_image(self, image_path: str) -> Image.Image:
        """Apply random augmentations to an image.

        Performs a series of random transformations including rotation, scaling,
        cropping, and brightness adjustment based on configured probabilities.

        Args:
            image_path (str): Path to the image file to augment

        Returns:
            Image.Image: Augmented image object, or None if augmentation fails

        Raises:
            IOError: If the image file cannot be opened
        """
        try:
            img = Image.open(image_path)

            # Randomly select augmentation operations
            if random.random() < self.image_aug_prob:
                # Random rotation
                if random.random() < 0.5:
                    angle = random.randint(-30, 30)
                    img = img.rotate(angle)

                # Random scaling
                if random.random() < 0.5:
                    scale = random.uniform(0.8, 1.2)
                    width, height = img.size
                    new_size = (int(width * scale), int(height * scale))
                    img = img.resize(new_size)

                # Random cropping
                if random.random() < 0.5:
                    width, height = img.size
                    left = random.randint(0, width // 4)
                    top = random.randint(0, height // 4)
                    right = width - random.randint(0, width // 4)
                    bottom = height - random.randint(0, height // 4)
                    img = img.crop((left, top, right, bottom))

                # Random brightness adjustment
                if random.random() < 0.5:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2))

            return img
        except Exception as e:
            print(f"Error augmenting image: {e}")
            return None

    def augment_text(self, text: str) -> str:
        """Apply random augmentations to text.

        Performs synonym replacement and random word deletion based on configured
        probabilities using a vocabulary file.

        Args:
            text (str): Input text to augment

        Returns:
            str: Augmented text, or original text if augmentation fails

        Raises:
            FileNotFoundError: If the vocabulary file is missing
        """
        try:
            if random.random() < self.text_aug_prob:
                # Initialize augmenters with vocab file path
                vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")

                # Synonym replacement
                if random.random() < 0.5:
                    substitute = WordSubstitute(aug_type="synonym", create_n=1, aug_n=1, vocab=vocab_path)
                    text = substitute.augment(text)[0]

                # Random deletion
                if random.random() < 0.5:
                    delete = WordDelete(create_n=1, aug_n=1, vocab=vocab_path)
                    text = delete.augment(text)[0]

            return text
        except Exception as e:
            print(f"Error augmenting text: {e}")
            return text

    def augment_samples(self, samples: List[Dict]) -> List[Dict]:
        """Augment a batch of samples containing image-text pairs.

        Args:
            samples (List[Dict]): List of samples where each sample contains:
                - image_path: Path to the image file
                - text: Associated text

        Returns:
            List[Dict]: List of augmented samples containing:
                - image_path: Original image path
                - text: Augmented text
                - augmented_image: Augmented image object

        Note:
            Samples that fail augmentation are excluded from the results
        """
        try:
            augmented_samples = []
            for sample in samples:
                # Image augmentation
                augmented_image = self.augment_image(sample["image_path"])
                if augmented_image is None:
                    continue

                # Text augmentation
                augmented_text = self.augment_text(sample["text"])

                # Create new sample
                augmented_samples.append(
                    {"image_path": sample["image_path"], "text": augmented_text, "augmented_image": augmented_image}
                )

            return augmented_samples
        except Exception as e:
            print(f"Error augmenting samples: {e}")
            return samples

    def augment(self, samples: List[Dict]) -> List[Dict]:
        """Main entry point for augmenting a dataset of image-text conversations.

        Args:
            samples (List[Dict]): List of samples where each sample contains:
                - id: Unique sample identifier
                - image: Image file name or path
                - conversations: List of conversation turns
                - meta: Optional metadata (preserved if present)

        Returns:
            List[Dict]: Combined list of original and augmented samples. Each
                augmented sample has a modified id with '_aug' suffix.

        Note:
            Invalid samples are skipped and not included in the results
        """
        try:
            augmented_samples = []

            for sample in samples:
                # Skip invalid samples
                if not all(k in sample for k in ["id", "image", "conversations"]):
                    continue

                # Get full image path
                image_path = self._get_image_path(sample["image"])

                # Image augmentation
                augmented_image = self.augment_image(image_path)
                if augmented_image is None:
                    continue

                # Text augmentation for each conversation
                augmented_conversations = []
                for conv in sample["conversations"]:
                    if "value" in conv:
                        augmented_text = self.augment_text(conv["value"])
                        augmented_conversations.append({"from": conv["from"], "value": augmented_text})

                # Create new augmented sample
                augmented_sample = {
                    "id": f"{sample['id']}_aug",
                    "image": sample["image"],
                    "conversations": augmented_conversations,
                }

                if "meta" in sample:
                    augmented_sample["meta"] = sample["meta"]

                augmented_samples.append(augmented_sample)

            # Combine original and augmented samples
            all_samples = samples + augmented_samples

            return all_samples

        except Exception as e:
            print(f"Error in augmentation pipeline: {e}")
            return samples
