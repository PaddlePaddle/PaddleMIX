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
anno_path = "datasets/llava/00_llava_v1_5_mix665k_convert.json"

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the filter operator
dataset = dataset.valid_data_filter()

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Dataset validation complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace(".json", "_base_filter.json"))
