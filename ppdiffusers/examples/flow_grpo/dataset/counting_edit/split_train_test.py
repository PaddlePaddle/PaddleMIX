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

import json
import random

# Load data from the JSONL file
data = []
with open("output.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Shuffle the data using random
random.seed(42)
random.shuffle(data)

# Split into test set (128 samples) and training set (remaining)
test_set = data[:112]
train_set = data[112:]

# Save the test set
with open("test_metadata.jsonl", "w") as f:
    for item in test_set:
        json.dump(item, f)
        f.write("\n")

# Save the training set
with open("train_metadata.jsonl", "w") as f:
    for item in train_set:
        json.dump(item, f)
        f.write("\n")

print(f"Test set size: {len(test_set)}")
print(f"Training set size: {len(train_set)}")
