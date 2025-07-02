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
import math
import os
import tempfile
import zipfile
from zipfile import ZipFile

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from paddlemix.appflow import Appflow

task = Appflow(
    app="auto_label",
    models=["paddlemix/blip2-caption-opt2.7b", "GroundingDino/groundingdino-swint-ogc", "Sam/SamVitH-1024"],
)


def auto_label(img, prompt):
    result = task(image=img, blip2_prompt=prompt)
    return result


def result2json(result, filename):
    label_data = {
        "version": "0.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": filename,
        "imageHeight": result["image"].size[1],
        "imageWidth": result["image"].size[0],
    }

    for i in range(len(result["labels"])):
        # label去掉末尾的置信度
        label = result["labels"][i]
        spl_idx = -1
        for j in range(len(label)):
            if label[j] == "(":
                spl_idx = j
        if spl_idx == -1:
            label = label
        else:
            label = label[:spl_idx]

        # 增加bbox
        rect = result["boxes"][i].tolist()
        xmin, ymin, xmax, ymax = rect
        label_data["shapes"].append(
            {
                "label": label,
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

        # 记录polygen
        seg_mask = result["seg_masks"][i].numpy()[0]
        mask_img = seg_mask.astype("uint8") * 255
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            for point in contour:
                points.append(point[0].tolist())

        # 增加polygen
        rect = result["boxes"][i]
        xmin, ymin, xmax, ymax = rect
        label_data["shapes"].append(
            {"label": label, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}}
        )

    return label_data


def generate_mask(img, result_masks):
    divide_part = int(255 / (math.ceil(len(result_masks) / 3) + 1))
    np_img = np.array(img)
    for i in range(len(result_masks)):
        color = [0, 0, 0]
        c = i % 3
        p = i // 3 + 1
        color[c] = divide_part * p
        mask = result_masks[i]
        M = mask.numpy()[0]
        np_img[M] = color
        print(color)
    img = Image.fromarray(np_img)
    return img


def al_fun(img, prompt):
    img = Image.fromarray(img.astype("uint8")).convert("RGB")
    result = auto_label(img, prompt)
    label_data = result2json(result, "tmpimg")
    # Draw BBox
    draw = ImageDraw.Draw(img)
    for i in range(len(result["boxes"])):
        rect = result["boxes"][i].tolist()
        draw.rectangle(rect, width=10)
    # Draw Mask
    mask_img = generate_mask(result["image"], result["seg_masks"])
    # Write File
    labeled_file = os.path.join(tmpdir, "labeled_date.json")
    with open(labeled_file, "w") as f:
        json.dump(label_data, f, indent=4)
    return img, mask_img, labeled_file


def al_file_fun(file_in, prompt):
    out_zip_file = os.path.join(tmpdir, "labeled.zip")
    with ZipFile(out_zip_file, "w") as zipObj:
        for _, imgname in enumerate(file_in):
            image_pil = Image.open(imgname.name)
            result = auto_label(image_pil, prompt)
            label_data = result2json(result, imgname.name.split("/")[-1])
            labeled_file = os.path.join(tmpdir, imgname.name.split("/")[-1] + ".josn")
            with open(labeled_file, "w") as f:
                json.dump(label_data, f, indent=4)
            zipObj.write(labeled_file)
    return out_zip_file


def al_zip_fun(zip_in, prompt):
    for _, zipname in enumerate(zip_in):
        with open("test.txt", "a") as f:
            f.write(zipname.name + "\n")
            f.write(zipname.name + "\n")
        zipfile.ZipFile(zipname.name).extractall(tmpdir)
        with open("test.txt", "a") as f:
            f.write("\n after extract \n")
    out_zip_file = os.path.join(tmpdir, "labeled.zip")
    with ZipFile(out_zip_file, "w") as zipObj:
        for root, _, files in os.walk(tmpdir, topdown=False):
            for name in files:
                if name.split(".")[-1] in ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]:
                    img_path = os.path.join(root, name)
                    json_path = os.path.join(root, name + ".json")

                    image_pil = Image.open(img_path)
                    result = auto_label(image_pil, prompt)
                    label_data = result2json(result, img_path)
                    with open(json_path, "w") as f:
                        json.dump(label_data, f, indent=4)
                    zipObj.write(json_path)
                    os.remove(img_path)
    return out_zip_file


with gr.Blocks() as demo:
    gr.Markdown("# 自动标注（AutoLabel）")
    with gr.Tab("单张图片标注"):
        with gr.Row():
            al_image_in = gr.Image(label="输入图片")
            al_image_out1 = gr.Image(label="BBox标注图片")
            al_image_out2 = gr.Image(label="Mask标注图片")
        al_text_in = gr.Text(label="Prompt", value="describe the image")
        al_file_out_ = gr.File(label="标注文件")
        al_button = gr.Button()
        al_button.click(
            fn=al_fun, inputs=[al_image_in, al_text_in], outputs=[al_image_out1, al_image_out2, al_file_out_]
        )
    with gr.Tab("上传多张图片批量标注"):
        with gr.Row():
            al_file_in = gr.Files(label="上传多张图片", file_types=[".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"])
            al_file_out = gr.File(label="标注结果")
        al_file_text_in = gr.Text(label="Prompt", value="describe the image")
        al_file_button = gr.Button()
        al_file_button.click(fn=al_file_fun, inputs=[al_file_in, al_file_text_in], outputs=[al_file_out])
    with gr.Tab("上传压缩包批量标注"):
        with gr.Row():
            al_zip_in = gr.Files(label="上传压缩包", file_types=[".zip"])
            al_zip_out = gr.File(label="标注结果")
        al_zip_text_in = gr.Text(label="Prompt", value="describe the image")
        al_zip_button = gr.Button()
        al_zip_button.click(fn=al_zip_fun, inputs=[al_zip_in, al_zip_text_in], outputs=[al_zip_out])


# for download file, use the tempfile
global tmpdir
with tempfile.TemporaryDirectory(dir=".") as tmpdir:
    demo.launch()
