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


from typing import Optional
from functools import partial
from ...core import MMDataset, register
import spacy

# python -m spacy download en_core_web_sm

# Load spaCy model
def load_spacy_model(lang: str):
    """
    Load the spaCy model based on the specified language.

    Args:
        lang (str): Language code, supports 'en' (English).

    Returns:
        spacy.Language: An instance of the spaCy language model.
    """
    if lang == 'en':
        return spacy.load("en_core_web_sm")  # English
    else:
        raise ValueError(f"Unsupported language: {lang}")


def is_entity_dependency_valid(item, nlp, min_dependency_num: int = 1, any_or_all: str = 'any') -> bool:
    """
    Check if the entity dependency relationships in the sample meet the specified conditions.

    Args:
        item (dict): A dictionary containing text information for the sample.
        nlp (spacy.Language): Loaded spaCy model.
        min_dependency_num (int): Minimum number of dependency edges per entity. Default is 1.
        any_or_all (str): Filtering strategy. 'any' means at least one entity must meet the condition,
                          'all' means all entities must meet the condition.

    Returns:
        bool: True if the entity dependencies meet the requirements, otherwise False.
    """
    # Get the text content and clean special characters
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Process the text using the spaCy model
    doc = nlp(user_conv)

    # Define rules for identifying entities
    entity_poss = ['NOUN', 'PROPN', 'PRON']  # Nouns, proper nouns, pronouns
    entity_tags = ['NN', 'NR', 'PN', 'NNS', 'NNP', 'NNPS', 'PRP']

    # Identify entities and initialize dependency counts
    entity_to_dependency_nums = {}
    for token in doc:
        if token.pos_ in entity_poss and token.tag_ in entity_tags:
            entity_to_dependency_nums[token] = 0

    # Count dependency edges for each entity
    for obj in entity_to_dependency_nums:
        if obj.dep_ != 'ROOT':  # Exclude root nodes
            entity_to_dependency_nums[obj] += 1

    for token in doc:
        # Skip punctuation
        if token.pos_ == 'PUNCT':
            continue

        # If the token's head is an entity, increment the dependency count
        if token.head in entity_to_dependency_nums.keys() and token.dep_ != 'ROOT':
            entity_to_dependency_nums[token.head] += 1

    # Get dependency counts for all entities
    dependency_counts = [n for _, n in entity_to_dependency_nums.items()]

    # Filtering logic
    if any_or_all == 'any':
        # At least one entity must meet the dependency condition
        return any(count >= min_dependency_num for count in dependency_counts)
    elif any_or_all == 'all':
        # All entities must meet the dependency condition
        return all(count >= min_dependency_num for count in dependency_counts)
    else:
        raise ValueError(f"Unsupported any_or_all value: {any_or_all}")


@register()
def text_entity_dependency_filter(
    dataset: MMDataset, 
    lang: str = 'en', 
    min_dependency_num: Optional[int] = 2, 
    any_or_all: str = 'any'
) -> MMDataset:
    """
    Filter the dataset based on entity dependency relationships in the samples.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        lang (str): Language of the text, supports 'en' (English).
        min_dependency_num (int): Minimum number of dependency edges per entity. Default is 1.
        any_or_all (str): Filtering strategy, 'any' means at least one entity must meet the condition,
                          'all' means all entities must meet the condition.

    Returns:
        MMDataset: The filtered dataset.
    """
    print(f"Filtering samples based on language {lang} and entity dependency condition ({any_or_all}), minimum dependency edges: {min_dependency_num}...")
    
    # Load the spaCy model (load once)
    nlp = load_spacy_model(lang)

    # Create the filter function
    filter_func = partial(is_entity_dependency_valid, nlp=nlp, min_dependency_num=min_dependency_num, any_or_all=any_or_all)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset