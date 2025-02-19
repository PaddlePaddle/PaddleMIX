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
import fasttext
import requests
from typing import Optional, List, Union
from functools import partial
from ...core import MMDataset, register

FASTTEXT_MODEL_PATH = "lid.176.bin"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Check and load the FastText model
def load_fasttext_model(model_path: str, model_url: str) -> fasttext.FastText._FastText:
    if not os.path.exists(model_path):
        print(f"FastText model file {model_path} not found. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"FastText model downloaded to {model_path}")
    print(f"Loading FastText model from {model_path}...")
    return fasttext.load_model(model_path)

# Check if the sample's language meets the requirements
def is_language_valid(item, lang: Optional[Union[str, List[str]]] = None, min_score: float = 0.8, lang_model: fasttext.FastText._FastText = None) -> bool:
    """
    Check if the sample's language matches the specified language(s) and has a confidence score above the minimum threshold.

    Args:
        item (dict): A sample containing text information.
        lang (Union[str, List[str], None]): Allowed language codes (a string, list of strings, or None).
        min_score (float): Minimum confidence score for the language. Default is 0.8.
        lang_model (fasttext.FastText._FastText): Loaded FastText model.

    Returns:
        bool: True if the sample's language matches the requirements and has a high enough confidence score; otherwise, False.
    """

    # Concatenate conversations into a single string for language detection
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>', '').replace('\n', '')

    try:
        prediction = lang_model.predict(user_conv, k=1)
        lang_id = prediction[0][0].replace("__label__", "")
        lang_score = prediction[1][0]
    except Exception as e:
        print(f"Language detection failed. Error: {e}")
        return False

    # Check language code and confidence score
    if lang:
        if isinstance(lang, str):
            lang = [lang]  # Convert single string to list
        return lang_id in lang and lang_score >= min_score
    else:
        # If no language is specified, only check confidence score
        return lang_score >= min_score

@register()
def language_id_filter(
    dataset: MMDataset, 
    lang: Optional[Union[str, List[str]]] = None, 
    min_score: float = 0.8,
) -> MMDataset:
    """
    Filter the dataset based on the language ID and confidence score of the samples.

    Args:
        dataset (MMDataset): Input dataset.
        lang (Union[str, List[str], None]): Allowed language codes (a string, list of strings, or None).
        min_score (float): Minimum confidence score for the language. Default is 0.8.

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples that do not meet the language ID requirements...")

    # Load the FastText model once
    lang_model = load_fasttext_model(FASTTEXT_MODEL_PATH, FASTTEXT_MODEL_URL)

    # Create the filter function
    filter_func = partial(is_language_valid, lang=lang, min_score=min_score, lang_model=lang_model)

    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )

    return filtered_dataset
