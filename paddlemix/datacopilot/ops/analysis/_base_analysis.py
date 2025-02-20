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
import json
import requests
from collections import Counter
from typing import Dict, Any

import fasttext
from paddlenlp.transformers import AutoTokenizer
from ...core import MMDataset, register, ParallelMode
from ..visualize._analysis_plot import visualize_results


FASTTEXT_MODEL_PATH = "lid.176.bin"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def load_fasttext_model(model_path: str = FASTTEXT_MODEL_PATH, model_url: str = FASTTEXT_MODEL_URL) -> fasttext.FastText._FastText:
    """
    Check and load the FastText language detection model. If the model file does not exist locally, it will be downloaded.

    Args:
        model_path (str): Path to the FastText model file. Default is "lid.176.bin".
        model_url (str): URL to download the FastText model if it is not found locally.

    Returns:
        fasttext.FastText._FastText: Loaded FastText model instance.
    """
    if not os.path.exists(model_path):
        print(f"FastText model file not found at {model_path}. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"FastText model successfully downloaded to {model_path}.")
        except Exception as e:
            print(f"Failed to download FastText model. Error: {e}")
            raise

    try:
        print(f"Loading FastText model from {model_path}...")
        return fasttext.load_model(model_path)
    except Exception as e:
        print(f"Failed to load FastText model from {model_path}. Error: {e}")
        raise


def detect_language_with_fasttext(text: str, lang_model) -> str:
    """
    Detect the language of a given text using FastText.

    Args:
        text (str): Input text for language detection.
        lang_model: Loaded FastText model.

    Returns:
        str: Detected language code (e.g., 'en', 'fr'). Returns "unknown" if detection fails.
    """
    try:
        prediction = lang_model.predict(text.strip(), k=1)
        return prediction[0][0].replace("__label__", "")
    except Exception:
        return "unknown"


def save_data_to_json(data: Any, filename: str):
    """
    Save data to a JSON file.

    Args:
        data (Any): Data to be saved.
        filename (str): Path to the JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def analyze_dataset_statistics(dataset: MMDataset) -> Dict[str, Any]:
    """
    Analyze the dataset and compute basic statistics, including valid and invalid items.

    Args:
        dataset (MMDataset): The dataset to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing dataset statistics.
    """
    valid_items = []
    invalid_count = 0

    for item in dataset.items:
        if "image" in item and isinstance(item.get("conversations"), list) and item["conversations"]:
            valid_items.append(item)
        else:
            invalid_count += 1

    conversation_counts = [
        len(item.get("conversations", [])) for item in valid_items
    ]
    total_conversations = sum(conversation_counts)
    max_conversations = max(conversation_counts, default=0)
    min_conversations = min(conversation_counts, default=0)
    avg_conversations = total_conversations / len(conversation_counts) if conversation_counts else 0

    unique_images = len(set(item.get("image", None) for item in valid_items if "image" in item))

    return {
        "total_records": len(dataset),
        "unique_images": unique_images,
        "total_conversations": total_conversations,
        "max_conversations": max_conversations,
        "min_conversations": min_conversations,
        "avg_conversations": avg_conversations,
        "invalid_item_count": invalid_count,
        "valid_items": valid_items,
    }


def analyze_language_distribution(dataset: MMDataset, lang_model) -> Dict[str, Any]:
    """
    Analyze the language distribution in the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.
        lang_model: Loaded FastText model.

    Returns:
        Dict[str, Any]: Language distribution and mismatched language statistics.
    """
    human_msgs, assistant_msgs = [], []
    languages = Counter()
    mismatched_language_pairs = 0
    mismatched_pairs = []

    def process_conversation(item):
        nonlocal mismatched_language_pairs
        for conv in item.get("conversations", []):
            if len(conv) < 2:
                continue

            human_text = conv[0]
            assistant_text = conv[1]

            human_msgs.append(human_text)
            assistant_msgs.append(assistant_text)

            human_lang = detect_language_with_fasttext(human_text, lang_model)
            assistant_lang = detect_language_with_fasttext(assistant_text, lang_model)

            if human_lang != "unknown" and assistant_lang != "unknown" and human_lang != assistant_lang:
                mismatched_language_pairs += 1
                mismatched_pairs.append({
                    "human_message": human_text,
                    "human_language": human_lang,
                    "assistant_message": assistant_text,
                    "assistant_language": assistant_lang
                })

            languages[human_lang] += 1
            languages[assistant_lang] += 1

    dataset.map(process_conversation, max_workers=8, mode=ParallelMode.THREAD, progress=True)

    return {
        "human_message_count": len(human_msgs),
        "assistant_message_count": len(assistant_msgs),
        "mismatched_language_pairs_count": mismatched_language_pairs,
        "languages_distribution": dict(languages),
    }


def analyze_image_paths(dataset: MMDataset) -> Dict[str, Any]:
    """
    Validate the distribution and existence of image paths in the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.

    Returns:
        Dict[str, Any]: Image path statistics and missing path details.
    """
    def extract_image_path(item):
        return item.get("image", None)

    all_paths = dataset.map(extract_image_path, max_workers=8, mode=ParallelMode.THREAD, progress=True)
    all_paths = [path for path in all_paths if path]
    missing_paths = [path for path in all_paths if not os.path.exists(path)]
    path_distribution = Counter(os.path.dirname(path) for path in all_paths)

    return {
        "total_images": len(all_paths),
        "missing_images": len(missing_paths),
        "path_distribution": dict(path_distribution),
    }



def analyze_data_anomalies(dataset: MMDataset, output_dir: str) -> Dict[str, int]:
    """
    Detect anomalies in the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.
        output_dir (str): Directory to save anomaly reports.

    Returns:
        Dict[str, int]: Counts of detected anomalies.
    """
    def identify_anomalies(item):
        # 初始化异常信息
        anomalies = {}

        # 检查 item 是否为有效数据
        if not isinstance(item, dict):  # 确保 item 是一个字典
            anomalies["invalid_item"] = True
            return anomalies

        # 检查是否缺少必须字段
        required_fields = ["image", "conversations"]
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            anomalies["missing_fields"] = missing_fields

        # 检查 conversations 是否为空或无效
        if "conversations" in item:
            if not item["conversations"]:  # conversations 为空
                anomalies["empty_conversations"] = True
            elif any(not conv[0].strip() for conv in item["conversations"]):  # conversations 内容无效
                anomalies["invalid_conversation_content"] = True

        return anomalies

    # 执行并行映射以分析数据集中的异常
    anomaly_results = dataset.map(identify_anomalies, max_workers=8, mode=ParallelMode.THREAD)

    # 分类异常
    missing_fields = [item for item, result in zip(dataset, anomaly_results) if result.get("missing_fields")]
    empty_conversations = [item for item, result in zip(dataset, anomaly_results) if result.get("empty_conversations")]
    invalid_items = [item for item, result in zip(dataset, anomaly_results) if result.get("invalid_item")]

    # 保存异常到 JSON 文件
    save_data_to_json(missing_fields, os.path.join(output_dir, "missing_fields.json"))
    save_data_to_json(empty_conversations, os.path.join(output_dir, "empty_conversations.json"))
    save_data_to_json(invalid_items, os.path.join(output_dir, "invalid_items.json"))

    # 返回异常统计
    return {
        "missing_field_count": len(missing_fields),
        "empty_conversation_count": len(empty_conversations),
        "invalid_item_count": len(invalid_items),
    }



def decode_token_ids(token_counts: Counter, tokenizer: AutoTokenizer) -> Counter:
    """
    Decode token IDs into their corresponding text and count their occurrences.

    Args:
        token_counts (Counter): A Counter object containing token IDs and their frequencies.
        tokenizer (AutoTokenizer): A tokenizer instance to decode the token IDs.

    Returns:
        Counter: A Counter object containing decoded tokens and their frequencies.
    """
    decoded_counts = Counter()
    for token_id, count in token_counts.items():
        try:
            decoded_text = tokenizer.decode([token_id]).strip()
            decoded_counts[decoded_text] += count
        except Exception as e:
            print(f"Error decoding token ID {token_id}: {e}")
    return decoded_counts


def analyze_single_conversation_tokens(item: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Analyze tokens in a single conversation.

    Args:
        item (Dict[str, Any]): A dictionary containing conversation data.
        tokenizer (AutoTokenizer): A tokenizer instance to tokenize the text.

    Returns:
        Dict[str, Any]: Token statistics for human and assistant messages.
    """
    human_tokens = []
    assistant_tokens = []

    for conv in item.get("conversations", []):
        try:
            # Extract human message tokens
            if len(conv) > 0:  # Ensure human message exists
                human_tokens.extend(
                    tokenizer(conv[0], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten()
                )
            # Extract assistant message tokens
            if len(conv) > 1:  # Ensure assistant message exists
                assistant_tokens.extend(
                    tokenizer(conv[1], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten()
                )
        except Exception as e:
            print(f"Error processing conversation: {conv}. Error: {e}")

    return {
        "human": {
            "total_tokens": len(human_tokens),
            "token_distribution": Counter(human_tokens),
        },
        "assistant": {
            "total_tokens": len(assistant_tokens),
            "token_distribution": Counter(assistant_tokens),
        }
    }


def analyze_conversation_tokens(dataset: MMDataset, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Perform token-level analysis on the dataset, including token distribution and frequency.

    Args:
        dataset (MMDataset): The dataset to analyze.
        tokenizer (AutoTokenizer): A tokenizer instance to tokenize the text in the dataset.

    Returns:
        Dict[str, Any]: Analysis results, including token distributions and high/low-frequency tokens.
    """
    print("Starting token analysis...")

    human_token_distribution = Counter()
    assistant_token_distribution = Counter()
    total_human_tokens = 0
    total_assistant_tokens = 0

    token_results = dataset.map(
        func=lambda item: analyze_single_conversation_tokens(item, tokenizer),
        max_workers=16,
        progress=True
    )

    for result in token_results:
        total_human_tokens += result["human"]["total_tokens"]
        total_assistant_tokens += result["assistant"]["total_tokens"]
        human_token_distribution.update(result["human"]["token_distribution"])
        assistant_token_distribution.update(result["assistant"]["token_distribution"])

    num_common_tokens = 20
    human_high_freq_tokens = human_token_distribution.most_common(num_common_tokens)
    assistant_high_freq_tokens = assistant_token_distribution.most_common(num_common_tokens)
    human_low_freq_tokens = human_token_distribution.most_common()[-num_common_tokens:]
    assistant_low_freq_tokens = assistant_token_distribution.most_common()[-num_common_tokens:]

    return {
        "human": {
            "total_tokens": total_human_tokens,
            "high_freq_tokens": decode_token_ids(Counter(dict(human_high_freq_tokens)), tokenizer),
            "low_freq_tokens": decode_token_ids(Counter(dict(human_low_freq_tokens)), tokenizer),
        },
        "assistant": {
            "total_tokens": total_assistant_tokens,
            "high_freq_tokens": decode_token_ids(Counter(dict(assistant_high_freq_tokens)), tokenizer),
            "low_freq_tokens": decode_token_ids(Counter(dict(assistant_low_freq_tokens)), tokenizer),
        }
    }


@register()
def base_analysis_pipeline(dataset: MMDataset, analysis_flags: Dict[str, bool] = None, output_dir: str = "output_directory") -> Dict[str, Any]:
    """
    Execute a pipeline of analysis functions on the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.
        analysis_flags (Dict[str, bool]): Flags to control which analyses to run.
        output_dir (str): Directory to save analysis results.

    Returns:
        Dict[str, Any]: Results of the analyses.
    """
    print("Initializing FastText model and Tokenizer...")
    lang_model = load_fasttext_model()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    if analysis_flags is None:
        analysis_flags = {
            "dataset_statistics": True,
            "language_distribution": True,
            "image_path_analysis": True,
            "data_anomalies": True,
            "conversation_tokens": True
        }

    results = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("analysis_flags:", analysis_flags)

    if analysis_flags.get("dataset_statistics", False):
        print("Running dataset statistics analysis...")
        results["dataset_statistics"] = analyze_dataset_statistics(dataset)

    if analysis_flags.get("language_distribution", False):
        print("Running language distribution analysis...")
        results["language_distribution"] = analyze_language_distribution(dataset, lang_model)

    if analysis_flags.get("image_path_analysis", False):
        print("Running image path validation...")
        results["image_path_analysis"] = analyze_image_paths(dataset)

    if analysis_flags.get("data_anomalies", False):
        print("Running anomaly detection...")
        results["data_anomalies"] = analyze_data_anomalies(dataset, output_dir)

    if analysis_flags.get("conversation_tokens", False):
        print("Running token analysis...")
        results["conversation_tokens"] = analyze_conversation_tokens(dataset, tokenizer)

    print("All analyses completed. Visualizing results...")
    visualize_results(results, output_dir, analysis_flags)

    return results