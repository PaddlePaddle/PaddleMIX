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
anno_path = "datasets/llava/02_val_chatml_filter.json"

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Analysis flags to specify which analyses to run
analysis_flags = {
    "dataset_statistics": False,
    "language_distribution": False,
    "image_path_analysis": True,
    "data_anomalies": False,
    "conversation_tokens": False,
}

# Run the base analysis
results = dataset.base_analysis_pipeline(analysis_flags=analysis_flags, output_dir="analysis_results")

# Print the results
print("Analysis results:", results)
