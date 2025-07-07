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
anno_path = "random_samples.json"

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the average line length filter
min_length = 15  # Set the minimum average line length
max_length = 50  # Set the maximum average line length
dataset = dataset.average_line_length_filter(min_length=min_length, max_length=max_length)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Average line length filtering completed.")

# Export the filtered dataset
dataset.export_json(anno_path.replace(".json", "_avg_line_length_filtered.json"))
