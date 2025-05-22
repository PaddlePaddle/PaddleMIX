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

import os
import json
import glob
import base64
from PIL import Image
from pathlib import Path
from functools import partial
from paddlemix.datacopilot.core import MMDataset
import re
import argparse

def read_json_files(json_root,image_root):
    # 创建空列表用于存储字典
    data_list = []

    for root, _, files in os.walk(json_root):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                image_path_png = json_path.replace('.json','.png').replace(json_root,image_root)
                image_path_jpg = json_path.replace('.json','.jpg').replace(json_root,image_root)

                 # 创建空字典用于存储数据
                data_dict = {}
                if os.path.exists(image_path_png):
                    image_path = image_path_png
                    data_dict['image'] = file.replace('.json','.png')
                else:
                    image_path = image_path_jpg
                    data_dict['image'] = file.replace('.json','.jpg')
                
                parent_path = os.path.dirname(image_root)
                data_dict['image']=image_path.replace(parent_path,'.')
                
                if os.path.exists(json_path) and os.path.exists(image_path):
                    # 打开并读取 JSON 文件内容
                    data_dict['conversations'] = MMDataset.from_json(json_path).items
                    # 将字典添加到列表中
                    data_list.append(data_dict)
                
                else:
                    print("File not found:", json_path)
    return data_list

def filter_image_hw_ratio(item,root):
    min_short_side = 20
    max_aspect_ratio = 200
    def check_image_size(image_path):
        try:
            with Image.open(image_path) as im:
                width, height = im.size
                short_side = min(width, height)
                long_side = max(width, height)
                aspect_ratio = long_side / short_side
                return short_side >= min_short_side and aspect_ratio <= max_aspect_ratio
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return False

    return check_image_size(os.path.join(os.path.dirname(root), item['image']))


def convert_scheme(item):
    #去除纯英文对话
    target = {"language": "en"}
    for it in item["conversations"]:
        if it == target:
            item["conversations"].remove(it)
            break
    images = []
    images.append(item['image'])

    #转化对话格式
    messages = []
    try:
        for conv in item['conversations']:
            messages.append({
                'content': conv['value'],
                'role': 'user' if conv['from'] == 'human' else 'assistant',
            })
    except Exception as e:
        print(item)
        print("Error processing item:", e)
    
    #检查 <image>数量
    for conv in messages:
        if conv['role'] == 'user' \
        and len(re.findall(r'<image>', conv['content'])) == 0 \
        and len(images) == 1:
            messages[0]['content'] = '<image>\n' + messages[0]['content']
            return dict(images=images, messages=messages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_root', type=str,required=True)
    parser.add_argument('--image_root', type=str,required=True)
    parser.add_argument('--output_file', type=str,
                        default=r'output_file.json')
    args = parser.parse_args()
    json_root = args.json_root
    image_root = args.image_root

    data_list = read_json_files(json_root,image_root)
    dataset = MMDataset(data_list)
    _filter_image_hw_ratio = partial(filter_image_hw_ratio, root=image_root)
    dataset = dataset.filter(_filter_image_hw_ratio).nonempty()
    dataset = dataset.map(convert_scheme).nonempty()
    print('数据长度',len(dataset))
    out_path = os.path.join(image_root, args.output_file)
    dataset.export_json(out_path)
    print(f"数据保存在 {out_path}")