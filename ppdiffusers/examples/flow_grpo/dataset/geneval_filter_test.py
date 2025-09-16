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


def filter_train_metadata(test_path, train_path, output_path):
    # 读取test文件中的所有prompt
    test_prompts = set()
    with open(test_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            test_prompts.add(data["prompt"])

    # 过滤train文件并写入新文件
    with open(train_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            data = json.loads(line.strip())
            if data["prompt"] not in test_prompts:
                fout.write(line)  # 保留原行，包括换行符


# 示例用法
if __name__ == "__main__":
    tasks = ["color_attr", "colors", "counting", "single_object", "two_object"]
    for task in tasks:
        filter_train_metadata(
            f"/dataset/geneval/{task}/test_metadata.jsonl",
            f"/dataset/geneval/{task}/train_metadata_nofiltered.jsonl",
            f"/dataset/geneval/{task}/train_metadata.jsonl",
        )
