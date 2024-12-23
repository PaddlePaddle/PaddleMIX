# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import io
import json
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--dataset_dir", type=str, default="pokemon_gpt4o_zh")
    parser.add_argument("--file_name", type=str, default="../data/pokemon_gpt4o_zh/pokemon_gpt4o_zh.parquet")

    args = parser.parse_args()
    return args


ROLE_MAPPING = {"human": "user", "gpt": "assistant"}


def to_conv_template(content, role):
    return {"role": role, "content": content}


def write_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    args = get_args()
    df = pd.read_parquet(args.file_name)
    output_path = os.path.join(args.data_dir, args.dataset_dir)
    converted_dataset = []
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    for i in tqdm(range(len(df))):
        new_conversation = {
            "messages": [],
            "images": [],
        }
        cur_conversation = df.iloc[i].conversations

        # process conversation
        for conv_id in range(len(cur_conversation)):
            role = cur_conversation[conv_id]["from"]
            role = ROLE_MAPPING[role]
            content = cur_conversation[conv_id]["value"]
            new_conversation["messages"] += [to_conv_template(content, role)]

        cur_images = df.iloc[i].images
        for img_id in range(len(cur_images)):
            save_dir = os.path.join(args.data_dir, "{}/images/{}_{}.png".format(args.dataset_dir, str(i), str(img_id)))
            if not os.path.exists(save_dir):
                img = Image.open(io.BytesIO(cur_images[img_id]["bytes"]))
                img.save(save_dir)
            new_conversation["images"].append("{}/images/{}_{}.png".format(args.dataset_dir, str(i), str(img_id)))
        # break
        converted_dataset += [new_conversation]
    if not os.path.exists(os.path.join(output_path, f"{args.dataset_dir}.json")):
        write_json(converted_dataset, os.path.join(output_path, "pokemon_gpt4o_zh.json"))

    # add to dataset info
    info_path = os.path.join(args.data_dir, "dataset_info.json")
    if os.path.exists(info_path):
        dataset_info = load_json(info_path)
    else:
        dataset_info = {}
    if args.dataset_dir not in dataset_info.keys():
        dataset_info[args.dataset_dir] = {
            "file_name": f"{args.dataset_dir}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
        write_json(dataset_info, os.path.join(args.data_dir, "dataset_info.json"))
