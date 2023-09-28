from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
import paddle
import cv2

import os
import json
from zipfile import ZipFile
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
import traceback

task = Appflow(app="auto_label",
               models=["paddlemix/blip2-caption-opt2.7b","GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"])

def auto_label(img, prompt):
    result = task(image=img,blip2_prompt = prompt)
    return result

def result2json(result, filename):
    label_data = {'version': '0.0.0',
                'flags': {} , 
                'shapes': [], 
                'imagePath': filename, 
                'imageHeight': result['image'].size[1], 
                'imageWidth': result['image'].size[0]}

    for i in range(len(result['labels'])):
        # label去掉末尾的置信度
        label = result['labels'][i]
        spl_idx = -1
        for j in range(len(label)):
            if label[j] == '(':
                spl_idx = j
        if spl_idx == -1:
            label = label
        else:
            label = label[:spl_idx]

        # 增加bbox
        rect = result['boxes'][i]
        xmin, ymin, xmax, ymax = rect
        label_data['shapes'].append(
            {'label': label,
            'points': [[xmin, ymin],[xmax, ymax]],
            'group_id': None,
            'shape_type': 'rectangle',
            'flags': {}
            }
        )
    
        # 记录polygen
        seg_mask = result['seg_masks'][i].numpy()[0]
        mask_img = seg_mask.astype('uint8')*255
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            for point in contour:
                points.append(point[0].tolist())

        # 增加polygen
        rect = result['boxes'][i]
        xmin, ymin, xmax, ymax = rect
        label_data['shapes'].append(
            {'label': label,
            'points': points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {}
            }
        )

    return label_data

def al_fun(img, prompt):
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    result = auto_label(img, prompt)
    label_data = result2json(result, " ")
    # 绘制
    draw = ImageDraw.Draw(img)
    for i in range(len(result['boxes'])):
        rect = result['boxes'][i].tolist()
        draw.rectangle(rect)
    return img, label_data

def al_file_fun(file_in, prompt):
    with ZipFile("labeled.zip", "w") as zipObj:
        for _, imgname in enumerate(file_in):
            image_pil = load_image(imgname.name)
            result = auto_label(image_pil, prompt)
            label_data = result2json(result, imgname.name.split("/")[-1])
            with open(imgname.name.split("/")[-1]+'.josn','w') as f:
                json.dump(label_data, f, indent=4)
            zipObj.write(imgname.name.split("/")[-1]+'.josn')
    return "labeled.zip"

def al_path_fun(path_in, prompt):
    with ZipFile("labeled.zip", "w") as zipObj:
        for root, _, files in os.walk(path_in, topdown=False):
            for name in files:
                if name.split('.')[-1] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
                    img_path = os.path.join(root, name)
                    json_path = os.path.join(root, name+'.json')

                    image_pil = load_image(img_path)
                    result = auto_label(image_pil, prompt)
                    label_data = result2json(result, img_path)
                    with open(json_path,'w') as f:
                        json.dump(label_data, f, indent=4)
                    zipObj.write(json_path)
    return "labeled.zip"


with gr.Blocks() as demo:
    gr.Markdown("# 自动标注（AutoLabel）")
    with gr.Tab("单张图片标注"):
        with gr.Row():
            al_image_in = gr.Image(label = "输入图片")
            al_image_out = gr.Image(label = "标注图片")
        al_text_in = gr.Text(label = "Prompt")
        al_text_out = gr.Text(label = "标注信息")
        al_button = gr.Button()
        al_button.click(fn=al_fun, inputs = [al_image_in, al_text_in], outputs = [al_image_out, al_text_out])
    with gr.Tab("上传文件批量标注"):
        with gr.Row():
            al_file_in = gr.Files(label = "上传多张图片", file_types=['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'])
            al_file_out = gr.File(label = "标注结果")
        al_file_text_in = gr.Text(label = "Prompt")
        al_file_button = gr.Button()
        al_file_button.click(fn=al_file_fun, inputs = [al_file_in, al_file_text_in], outputs = [al_file_out])
    with gr.Tab("指定路径下批量标注"):
        al_path_in = gr.Text(label = "待标注图片所在目录")
        al_path_text_in = gr.Text(label = "Prompt")
        al_path_out = gr.File(label = "标注结果")
        al_path_button = gr.Button()
        al_path_button.click(fn=al_path_fun, inputs = [al_path_in, al_path_text_in], outputs = [al_path_out])

demo.launch()
