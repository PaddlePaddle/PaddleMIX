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


from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from typing import Dict
from ...core import MMDataset, register
from ...misc import parallel_map, ParallelMode


def preprocess_text(text: str) -> str:
    """
    Cleans placeholder `<image>` and extra newlines from the text.

    Args:
        text (str): Original text.

    Returns:
        str: Cleaned text.
    """
    return text.replace("<image>", "").replace("\n<image>", " ").replace("<image>\n", " ").strip()


def simhash_duplicate_operator(
    text: str,
    seen_hashes: set,
    threshold: float = 0.8
) -> bool:
    """
    Checks for duplicate text using the SimHash algorithm.

    Args:
        text (str): Input text.
        seen_hashes (set): Set of previously recorded SimHash values.
        threshold (float): Similarity threshold for Hamming distance (default: 0.8).

    Returns:
        bool: True if the text is a duplicate, False otherwise.
    """
    simhash_value = Simhash(text).value
    for existing_hash in seen_hashes:
        # Calculate Hamming distance
        distance = bin(simhash_value ^ existing_hash).count("1")
        if distance <= int((1 - threshold) * 64):  # SimHash length is 64
            return True
    seen_hashes.add(simhash_value)
    return False


def minhash_duplicate_operator(
    text: str,
    lsh: MinHashLSH,
    num_perm: int = 128,
    counter: list = [0], 
) -> bool:
    """
    Checks for duplicate text using the MinHashLSH algorithm.

    Args:
        text (str): Input text.
        lsh (MinHashLSH): MinHashLSH instance.
        num_perm (int): Number of hash functions for MinHash.
        counter (list): Counter to generate unique keys for MinHashLSH entries.

    Returns:
        bool: True if the text is a duplicate, False otherwise.
    """
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))

    # Query the LSH for similar entries
    if list(lsh.query(minhash)):
        return True

    # Generate a unique key and insert the MinHash
    unique_key = f"key_{counter[0]}"
    counter[0] += 1
    lsh.insert(unique_key, minhash)
    return False


@register()
def conversation_hash_filter(
    dataset: MMDataset,
    method: str = "simhash",
    threshold: float = 0.8,
    num_perm: int = 128,
) -> MMDataset:
    """
    Removes duplicate Q&A pairs in conversations using either SimHash or MinHashLSH.

    Args:
        dataset (MMDataset): Input dataset.
        method (str): Deduplication method, either 'simhash' or 'minhash'.
        threshold (float): Similarity threshold (for SimHash, it is Hamming distance ratio).
        num_perm (int): Number of hash functions for MinHash (only used for MinHash).

    Returns:
        MMDataset: Dataset after deduplication.
    """
    if method not in {"simhash", "minhash"}:
        raise ValueError("Unsupported method. Choose 'simhash' or 'minhash'.")

    # Initialize counters
    total_pairs = 0
    removed_pairs = 0

    def filter_unique_conversations(item: Dict) -> Dict:
        """
        Processes each conversation and removes duplicate Q&A pairs.
        """
        nonlocal total_pairs, removed_pairs

        local_seen_hashes = set()
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if method == "minhash" else None
        unique_conversations = []

        for question, answer in item.get("conversations", []):
            text = f"{preprocess_text(question)} {preprocess_text(answer)}"
            total_pairs += 1
            if method == "simhash":
                if simhash_duplicate_operator(text, local_seen_hashes, threshold):
                    removed_pairs += 1
                else:
                    unique_conversations.append([question, answer])
            elif method == "minhash":
                if minhash_duplicate_operator(text, lsh, num_perm):
                    removed_pairs += 1
                else:
                    unique_conversations.append([question, answer])

        if unique_conversations:
            return {"image": item["image"], "conversations": unique_conversations}
        return None

    # Process conversations in parallel
    filtered_items = parallel_map(
        filter_unique_conversations,
        dataset.items,
        max_workers=8,
        mode=ParallelMode.THREAD,
        progress=True,
        order=False,
    )

    # Output statistics
    retained_pairs = total_pairs - removed_pairs
    print(f"Total Q&A pairs: {total_pairs}")
    print(f"Filtered Q&A pairs: {removed_pairs}")
    print(f"Remaining Q&A pairs: {retained_pairs}")

    return MMDataset(filtered_items)