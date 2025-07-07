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
import json
import os

import paddle
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="vision tokenizer path")
    parser.add_argument("--data-path", type=str, help="data path")
    parser.add_argument("--output-path", type=str, help="tokenized data save path")
    parser.add_argument("--image-area", type=int, default=720 * 720)
    args = parser.parse_args()
    return args


def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5
    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))
    image = image.resize((tw, th))
    return image


def main():
    args = prepare_args()
    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path, device_map="cuda:0")
    image_tokenizer.eval()
    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)
    datalist = {"prefix": f"{args.output_path}/feature", "path_list": []}
    with open(args.data_path) as f:
        input_data = json.load(f)
    for inp in input_data:
        name = inp["name"]
        prompt = inp["text"]
        image = Image.open(inp["image"]).convert("RGB")
        image = smart_resize(image, args.image_area)
        image = image_processor(image, return_tensors="pt")["pixel_values"]
        with paddle.no_grad():
            image = image.cuda(blocking=True)
            token_ids = image_tokenizer.encode(image)
        token_ids = token_ids.squeeze(axis=0).cpu().numpy()
        data = {"name": name, "images": token_ids, "texts": prompt}
        paddle.save(obj=data, path=f"{args.output_path}/feature/{name}.pth")
        datalist["path_list"].append(f"{name}.pth")
    with open(f"{args.output_path}/list/train.json", "w") as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    main()
