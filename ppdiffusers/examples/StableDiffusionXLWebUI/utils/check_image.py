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
import time

from PIL import Image, PngImagePlugin


def custom_save_image(image, metadata, root, model_dir, *args):
    output_dir = f"{root}/{model_dir.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    extra = ("_".join(args) + "_") if args else ""
    filename = f"{extra}{time.strftime('%Y%m%d_%H%M%S', time.localtime())}{time.time()}.png"
    outpath = os.path.join(output_dir, filename)

    # 创建PngInfo对象
    png_info = PngImagePlugin.PngInfo()
    # 添加文本元数据
    for key, value in metadata.items():
        if not isinstance(value, str):
            value = str(value)
        png_info.add_text(key, value)
    # metadata_str = json.dumps(metadata)
    # png_info.add_text("metadata", metadata_str)
    image.save(outpath, "PNG", pnginfo=png_info)


def check_image_infos(image_path_or_file):
    if isinstance(image_path_or_file, Image.Image):
        image = image_path_or_file
    else:
        with Image.open(image_path_or_file) as image:
            image = image
    try:
        png_info = image.pnginfo  # 注意：这通常不是一个标准的属性
        return png_info.text
    except AttributeError:
        # print('No PNGInfo found in the image object.')
        return image.info
