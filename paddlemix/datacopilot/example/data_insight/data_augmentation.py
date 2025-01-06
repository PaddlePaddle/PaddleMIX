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

import random
from typing import Dict, List

from paddlenlp.dataaug import WordDelete, WordSubstitute
from PIL import Image, ImageEnhance


class DataAugmentor:
    def __init__(self):
        self.image_aug_prob = 0.5  # Image augmentation probability
        self.text_aug_prob = 0.3  # Text augmentation probability

    def augment_image(self, image_path: str) -> Image.Image:
        """Image augmentation"""
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
        """Text augmentation"""
        try:
            if random.random() < self.text_aug_prob:
                # Synonym replacement
                if random.random() < 0.5:
                    substitute = WordSubstitute()
                    text = substitute(text)

                # Random deletion
                if random.random() < 0.5:
                    delete = WordDelete()
                    text = delete(text)

            return text
        except Exception as e:
            print(f"Error augmenting text: {e}")
            return text

    def augment_samples(self, samples: List[Dict]) -> List[Dict]:
        """Sample augmentation"""
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
        """Main entry point for data augmentation"""
        try:
            augmented_samples = []

            for sample in samples:
                # Skip invalid samples
                if not all(k in sample for k in ["id", "image", "conversations"]):
                    continue

                # Image augmentation
                augmented_image = self.augment_image(sample["image"])
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
