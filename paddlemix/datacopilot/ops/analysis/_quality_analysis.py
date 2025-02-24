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

from tqdm import tqdm
from typing import Dict

import paddle
from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import Qwen2VLImageProcessor, Qwen2VLProcessor, process_vision_info
from ...core import MMDataset, register

# Predefined evaluation metrics and corresponding prompt templates
CRITERIA_PROMPTS = {
    "image_text_matching": """Please evaluate if the provided text caption accurately represents the main features and objects of the image. The caption doesn't need to detail every aspect of the image, but it should capture its primary theme. Rate the overall quality of the text caption's match to the image on a scale of 1-100, considering the criteria mentioned.""",
    "object_detail_fulfillment": """Please evaluate the text caption to determine if it provides detailed descriptions of objects that align with the image. Specifically, assess if the caption sufficiently describes the color, size, position, shape, material, etc., of the objects. Afterward, rate the caption's overall accuracy in capturing object details from the image on a scale of 1-100, based on the criteria provided.""",
    "caption_text_quality": """Please evaluate the text caption based on the following criteria: Grammatical Correctness, Diversity of Vocabulary (e.g., the range and uniqueness of words used), Fluency (e.g., smoothness and natural flow of sentences), Readability, Length, and Structure. Assign an overall quality score on a scale of 1-100.""",
    "semantic_understanding": """Evaluate the given text caption in relation to its corresponding image. Your goal is to determine if the text caption provides additional semantic information that isn't readily apparent just from the image itself. Rate the text caption's semantic depth on a scale from 1 to 100.""",
}

DEFAULT_PROMPT_TEMPLATE = """Text Caption: {caption}

{criteria}
A higher score indicates a higher level of {aspect}. Ensure that your scoring is nuanced and uses the entire range from 0 to 100, reflecting the subtle differences. The score should be given as an integer, with each number between 0 and 100 considered as a potential score, avoiding the tendency to round to multiples of 10. Please first output a single line containing the value indicating the score. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

# Load the Qwen2-VL-7B-Instruct model and processor
def load_model(model_name: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, dtype="bfloat16")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
    image_processor = Qwen2VLImageProcessor()
    processor = Qwen2VLProcessor(image_processor, tokenizer)
    return model, processor

def parse_model_output(output: str) -> int:
    """
    Parse the model's output to extract the score.

    Args:
        output (str): The raw output from the model.

    Returns:
        int: The extracted score (integer between 0 and 100).
    """
    lines = output.strip().split('\n')
    for line in lines:
        # Try to find the first line that contains a valid integer score
        try:
            # Remove any potential leading/trailing whitespace
            score_str = line.strip()
            # Attempt to convert to integer
            score = int(score_str)
            # Check if the score is within the valid range
            if 0 <= score <= 100:
                return score
        except ValueError:
            # Not a valid integer, move to the next line
            continue
    # If no valid score is found, return a default value or raise an error
    return -1  # -1 indicates an invalid or missing score

def evaluate_image_caption(
    dataset: MMDataset,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    analysis_flags: Dict[str, bool] = None
) -> Dict:
    """
    Evaluate the quality of image captions based on predefined metrics.

    Args:
        dataset (MMDataset): The dataset containing image paths and conversations.
        model_name (str): Name of the model to use.
        analysis_flags (Dict[str, bool]): Flags to control which metrics to evaluate.

    Returns:
        Dict: Evaluation results for each dataset item.
    """
    # Load the model and processor
    model, processor = load_model(model_name)

    # Final results storage
    results = {}

    # Determine which metrics to evaluate based on analysis_flags
    if analysis_flags is None:
        selected_metrics = list(CRITERIA_PROMPTS.keys())  # Default: All metrics
    else:
        selected_metrics = [key for key, value in analysis_flags.items() if value]

    # Process each item in the dataset
    for item in tqdm(dataset):
        item_id = item["image"]  # Use image path as item_id
        conversations = item["conversations"]
        print(item_id)

        # Combine all Q&A pairs into a single conversation
        full_caption = ""
        for conversation in conversations:
            question, answer = conversation
            # Combine question and answer into the caption
            full_caption += f"Question: {question}\nAnswer: {answer}\n"

        # Prepare image input
        image_inputs, video_inputs = process_vision_info([
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item_id,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ])

        # Evaluate each selected metric
        for metric in selected_metrics:
            criteria = CRITERIA_PROMPTS[metric]
            aspect = metric.replace("_", " ")
            caption = full_caption
            print(f"metric:{metric}, caption:{caption}")

            # Generate the full prompt
            full_prompt = DEFAULT_PROMPT_TEMPLATE.format(
                caption=caption,
                criteria=criteria,
                aspect=aspect
            )

            # Combine instruction and question
            image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
            text = f"<|im_start|>system\n{full_prompt}<|im_end|>\n<|im_start|>user\n{image_pad_token}<|im_end|>\n<|im_start|>assistant\n"

            # Tokenize and process inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pd",
            )

            # Generate output
            with paddle.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)
                decoded_output = processor.batch_decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                # Extract the score from the model's output
                score = parse_model_output(decoded_output[0])
                print(decoded_output[0])
                print("*"*50)

            # Store results (only the score)
            if item_id not in results:
                results[item_id] = {}
            results[item_id][metric] = score

    return results

@register()
def quality_analysis(
    dataset: MMDataset,
    model_name: str,
    quality_analysis_flags: Dict[str, bool] = None
):
    """
    Analyze the quality of multi-turn conversations for image captioning.

    Args:
        dataset (MMDataset): The dataset containing image paths and conversations.
        model_name (str): Name of the model to use.
        quality_analysis_flags (Dict[str, bool]): Flags to control which metrics to evaluate.

    Returns:
        Dict: Evaluation results for each dataset item.
    """
    results = evaluate_image_caption(
        dataset,
        model_name,
        quality_analysis_flags
    )
    return results