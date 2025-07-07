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

from paddlemix.datacopilot.core import MMDataset

# Path to the dataset
anno_path = "random_samples_1w.json"

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the conversation_hash_filter operator
dataset = dataset.conversation_hash_filter(
    method="simhash",  # Use the 'simhash' method (default)
    threshold=0.8,  # Similarity threshold
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Text deduplication complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace(".json", "_conversation_hash_filter.json"))
