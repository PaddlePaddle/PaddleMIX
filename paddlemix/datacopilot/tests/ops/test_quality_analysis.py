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

# Analysis flags to specify which analyses to run
quality_analysis_flags = {
    "image_text_matching": True,
    "object_detail_fulfillment": True,
    "caption_text_quality": True,
    "semantic_understanding": True,
}

# Apply the image caption metrics analysis operator
dataset_results = dataset.quality_analysis(
    model_name="Qwen/Qwen2-VL-7B-Instruct",  # Specify the model name
    quality_analysis_flags=quality_analysis_flags,  # Pass the analysis flags
)

# Print the results of the evaluation
print("Evaluation results:", dataset_results)
