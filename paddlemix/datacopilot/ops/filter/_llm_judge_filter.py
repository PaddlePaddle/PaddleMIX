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

import re
from tqdm import tqdm
from typing import Dict, List

import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

from ...core import MMDataset, register


def load_model(model_name: str):
    """
    Load the tokenizer and model for the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

def extract_rating(text: str) -> int:
    """
    Extract the Total rating value from the text.
    Returns -1 if no valid rating is found.
    """
    match = re.search(r'Total rating:\s*([1-4])(?:\D|$)', text)
    if match:
        return int(match.group(1))
    return -1

prompt_template = """
You will be given a user_question and system_answer couple. Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question. Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question. Here is the scale you should use to build your answer: 
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial 
2: The system_answer is mostly not helpful: misses some key aspects of the question 
3: The system_answer is mostly helpful: provides support, but still could be improved 
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question 
Provide your feedback as follows: 
Feedback::: 
Evaluation: (your rationale for the rating, as a text) 
Total rating: (your rating, as a number between 1 and 4) 
You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer. 
Now here are the question and answer. 
Question: {question} 
Answer: {answer} 
Provide your feedback.
"""

@register()
def llm_judge_filter(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-7B", batch_size: int = 1) -> Dict:
    """
    Analyze and filter Q&A pairs in the dataset using a llm model.

    Args:
        dataset (MMDataset): Input dataset containing Q&A pairs.
        model_name (str): Model name for the llm model (default: "Qwen/Qwen2.5-7B").
        batch_size (int): Batch size for processing (default: 1).

    Returns:
        MMDataset: Filtered dataset with Q&A pairs having a rating >= 3.
    """
    tokenizer, model = load_model(model_name)
    model.eval()
    
    all_data = []  # Store all Q&A pairs
    total_pairs = 0
    valid_pairs = 0
    filtered_data = {}  # Store filtered data organized by image path

    print("Collecting data...")
    for item in dataset:
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])
        
        for conv in conversations:
            if isinstance(conv, list) and len(conv) == 2:
                total_pairs += 1
                question, answer = conv
                cleaned_question = question.strip()
                
                all_data.append({
                    'image_path': image_path,
                    'question': cleaned_question,
                    'answer': answer,
                    'prompt': prompt_template.format(question=cleaned_question, answer=answer)
                })

    total_samples = len(all_data)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"Collected {total_samples} Q&A pairs, processing in {num_batches} batches.")

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        
        batch_data = all_data[start_idx:end_idx]
        batch_prompts = [item['prompt'] for item in batch_data]

        try:
            input_features = tokenizer(batch_prompts, return_tensors="pd", padding=True)
            
            with paddle.no_grad():
                outputs = model.generate(**input_features, max_length=128)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if not isinstance(outputs, paddle.Tensor):
                    outputs = paddle.to_tensor(outputs)
                
                outputs_list = outputs.numpy().tolist()
                
            decoded_outputs = tokenizer.batch_decode(outputs_list, skip_special_tokens=True)

            # Process results for the current batch
            for idx, eval_result in enumerate(decoded_outputs):
                
                rating = extract_rating(eval_result)
                current_item = batch_data[idx]
                print(f"Current Q&A pair: {current_item}, Rating: {rating}")
                # Keep Q&A pairs with a rating >= 3
                if rating >= 3:
                    valid_pairs += 1
                    image_path = current_item['image_path']
                    
                    if image_path not in filtered_data:
                        filtered_data[image_path] = {
                            "image": image_path,
                            "conversations": []
                        }
                    
                    filtered_data[image_path]["conversations"].append([
                        current_item['question'],
                        current_item['answer']
                    ])

        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}/{num_batches}:")
            print(f"Error details: {e}")
            print("-" * 50)
            continue

    final_dataset = list(filtered_data.values())

    print("Processing complete:")
    print(f"Total Q&A pairs: {total_pairs}")
    print(f"Valid Q&A pairs (rating >= 3): {valid_pairs}")
    print(f"Validity rate: {(valid_pairs/total_pairs*100):.2f}%")
    print(f"Number of images involved: {len(filtered_data)}")

    return MMDataset(final_dataset)