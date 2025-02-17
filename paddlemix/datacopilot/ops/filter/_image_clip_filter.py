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
from tqdm import tqdm
from typing import Optional, List
from dataclasses import dataclass

import paddle
from PIL import Image, ImageDraw, ImageFont
from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
from paddlemix.models.clip.clip_model import CLIP
from paddlemix.processors.tokenizer import SimpleTokenizer
from paddlemix.datacopilot.core import MMDataset, register
from ...misc import parallel_map, ParallelMode


@dataclass
class CLIPFilterConfig:
    """Configuration for CLIP filtering."""
    model_name: str = "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"
    threshold: float = 0.25
    batch_size: int = 8  # Batch size
    save_images: bool = False  # Whether to save low-confidence images
    save_dir: str = "./low_confidence_images"  # Directory to save low-confidence images


def clip_process_batch(
    image_paths: List[str],
    text_prompts: List[str],
    model: CLIP,
    processor: CLIPProcessor,
) -> List[Optional[float]]:
    """Processes a batch of images and texts, returning similarity scores."""
    try:
        processed_inputs = processor(
            images=image_paths,
            text=text_prompts,
            max_length=77,
            return_tensors="pd",
            return_attention_mask=False,
            mode="eval",
            do_resize=True,
            do_crop=True,
            padding_zero=True,
        )

        image_tensor = processed_inputs["image"]
        input_ids = processed_inputs["input_ids"]

        with paddle.no_grad():
            similarities = model.clip_score(image=image_tensor, input_ids=input_ids)

        return [float(similarity.item()) for similarity in similarities]
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return [None] * len(image_paths)


def clean_question(question: str) -> str:
    """Cleans the question text by removing `<image>` placeholders and extra newlines."""
    return question.replace("<image>", "").replace("\n<image>", " ").replace("<image>\n", " ").strip()


def format_text(question: str, answer: str) -> str:
    """Formats the text as 'question: ... \nanswer: ...'."""
    return f"question: {question} \nanswer: {answer}"


def contains_coordinates(text: str) -> bool:
    """Check if the text contains a 4-coordinate format [x1, y1, x2, y2]."""
    pattern = r"\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\]"
    return bool(re.search(pattern, text))


def save_combined_image(image_path, text, similarity, save_dir, sample_index):
    """Combines the original image and the Q&A pair + similarity score into a single image and saves it."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        # Define font and text area dimensions
        font = ImageFont.load_default()
        text_height = 100
        combined_width = image_width
        combined_height = image_height + text_height

        # Create a combined image
        combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined_image.paste(image, (0, 0))

        # Draw text
        draw = ImageDraw.Draw(combined_image)
        text_area_y = image_height + 10
        draw.text((10, text_area_y), text, fill="black", font=font)
        draw.text((10, text_area_y + 40), f"Similarity: {similarity:.2f}", fill="black", font=font)

        # Generate save path
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"low_confidence_{sample_index}.jpg")

        # Save the image
        combined_image.save(save_path)
        print(f"Saved low-confidence sample: {save_path}")
    except Exception as e:
        print(f"Error saving combined image: {e}")


@register()
def image_clip_filter(
    dataset: MMDataset,
    config: Optional[CLIPFilterConfig] = None
) -> MMDataset:
    """Filters out low-confidence Q&A pairs using CLIP and optionally saves the images."""
    if config is None:
        config = CLIPFilterConfig()
    save_dir = config.save_dir

    model = CLIP.from_pretrained(config.model_name, ignore_mismatched_sizes=False)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    filtered_items = []
    batch_size = config.batch_size

    all_samples = []
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        conversations = []
        for conversation in item.get("conversations", []):
            question, answer = conversation
            if contains_coordinates(question) or contains_coordinates(answer):
                continue  # Skip question-answer pairs that contain coordinates
            cleaned_question = clean_question(question)
            formatted_text = format_text(cleaned_question, answer)
            conversations.append((image_path, formatted_text, conversation))
        all_samples.extend(conversations)

    sample_index = 0
    low_confidence_samples = []
    for i in tqdm(range(0, len(all_samples), batch_size), desc="Filtering low-confidence Q&A pairs"):
        batch = all_samples[i:i + batch_size]
        image_paths = [sample[0] for sample in batch]
        text_prompts = [sample[1] for sample in batch]

        similarities = clip_process_batch(
            image_paths=image_paths,
            text_prompts=text_prompts,
            model=model,
            processor=processor,
        )
        for (image_path, formatted_text, conversation), similarity in zip(batch, similarities):
            if similarity is not None and similarity < config.threshold:
                if config.save_images:
                    save_combined_image(
                        image_path=image_path,
                        text=formatted_text,
                        similarity=similarity,
                        save_dir=save_dir,
                        sample_index=sample_index,
                    )
                sample_index += 1
                low_confidence_samples.append((image_path, conversation))

    def filter_high_confidence(item):
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            return None
        new_conversations = [
            conversation for conversation in item.get("conversations", [])
            if (image_path, conversation) not in low_confidence_samples
        ]
        if new_conversations:
            return {"image": image_path, "conversations": new_conversations}
        return None

    filtered_items = parallel_map(
        filter_high_confidence,
        dataset.items,
        max_workers=8,
        mode=ParallelMode.THREAD,
        progress=True,
        order=False,
    )

    return MMDataset(filtered_items)