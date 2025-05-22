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

from distutils.command.config import config
import os
import multiprocessing
from paddlex import create_pipeline
import json  
import re
import argparse
import erniebot
 
def initialize_model(access_token="", ak="", sk="", api_type="aistudio", max_retries=2):
    config = {
        "api_type": api_type,
        "max_retries": max_retries,
    }
    if access_token:
        config["access_token"] = access_token
    else:
        config["ak"] = ak
        config["sk"] = sk
    return config
 
def predict(prompts, model_name, config, temperature=0.001):
    chat_completion = erniebot.ChatCompletion.create(
        _config_=config,
        model=model_name,
        messages=[{"role": "user", "content": prompts}],
        temperature=float(temperature),
    )
    res = chat_completion.get_result()
    return res

template = '''
```json
[
    {
        "from": "human",
        "value": "xxx"
    },
    {
        "from": "gpt",
        "value":  "xxx"
    },
    {
        "from": "human",
        "value": "xxx"
    },
    {
        "from": "gpt",
        "value":  "xxx"
    }
]
```
'''
# generate_qa_pairs(file_path, res_json, json_output_path)
def generate_qa_pairs(image_path, json_string, qa_json_path, config, args):
    if os.path.exists(qa_json_path):
        print(qa_json_path,'already exists!!')
        return
    
    try:
        system = "你是一个文档数据视觉问答对话生成系统，主要任务是基于提供的文档图片的ocr版面信息，设计指令和对应答案，以便得到的多模态数据用于模型训练时，模型能够充分学习多模态文档理解能力。"
        question = f'''这里是从一个研报文档图片中提取的ocr版面信息：{json_string}，请根据其中的内容想象你正在看对应的图片而不是简单的json字符串（文本相关字符是按行识别的，请完整拼接后进行理解），然后结合图片中的信息从专业研报阅读者关心的角度设计指令和回答。
        在生成这些对话时，请确保遵循以下指导原则：
        # 这些指令必须满足以下的要求：
        ```
        1. **指令重点关注文档图片中的信息抽取能力，答案可以直接从图中观察得到。**
        2. **指令和回答必须要与图片高度相关，指令无法脱离图片进行回答。如果指令中提供了太多图片中的信息，使得可以脱离图片对指令进行回答，那是坚决不允许的。**
        3. **指令必须根据图片中的信息生成，且可以结合图片内容得到准确的答案**
        4. **提供的ocr版面信息中的文本字符包含了回答的确切答案，请严格确保回答正确，否则不要生成该指令和回答**
        5. **对于每个版面区域类型（印刷文字、表格、图表、印刷公式、印章），如果文档中包含该类型的信息，那么请生成基于这个类型内容的指令，否则不需要生成**
        6. **指令的回答尽可能简洁且准确，不要重复问题，直接回复答案，同时保证回答直接从图片原文中获取，不要进行总结概括**
        7. **指令与问答中不要出现有关版面结构的信息。**
        8. **指令直接给出问题，不要出现‘请问'、'请回答'、'在文档中'等字眼**
        9.**指令的回答请用图片中完整出现的内容回答，可以是单词、短语或句子，针对问题回答尽可能详细和完整，并保持格式、单位、符号和标点都与图片中的文字内容完全一致。**
        ```
        请生成至少5个中文指令和回答。你需要通过JSON格式提供生成内容，请确保输出中包含```json ```，可以参考下面的样例组织你的输出：{template}，如果文档中的内容全部为英文，则在生成的结果中添加一个键值对："language":"en"。
        '''
        assistant_reply = predict(prompts=system+question, model_name=args.ernie_model_name, config=config)
        print(assistant_reply)
    except Exception as e:
        print(f'处理图片{image_path}失败!!!!!:{e}')
        return
    try:
        json_spec_blocks = re.findall(
            r'```(.*?)```', assistant_reply, re.DOTALL)
        if json_spec_blocks:
            json_spec = json_spec_blocks[0].strip()
        else:
            json_spec = assistant_reply
        json_spec = json_spec.lstrip('json')
        qa_pairs = json.loads(json_spec)
        with open(qa_json_path, 'w') as file:
            json.dump(qa_pairs, file, indent=4,ensure_ascii=False)
        print(f"Save qa_pairs to {qa_json_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def set_img_to_empty(d):
    for k, v in d.items():
        if isinstance(v, dict):
            set_img_to_empty(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    set_img_to_empty(item)
                else:
                    if k == 'img':
                        d[k] = None
        else:
            if k == 'img':
                d[k] = None

def process_image(file_path, input_dir, output_dir, pipeline, config, args):
    # 预测
    output = pipeline.predict(file_path)

    # 生成输出目录路径
    relative_path = os.path.relpath(file_path, input_dir)
    json_output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.json')

    if os.path.exists(json_output_path):
        print(f'File already exists: {json_output_path}')
        return
    # 创建输出文件夹
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # 保存结果
    try:
        n =0
        for res in output:
            res_json = res.json['layout_parsing_result']
            set_img_to_empty(res_json) #不包含图片的OCR版面信息
            
            generate_qa_pairs(file_path, res_json, json_output_path, config, args)

        print(f"Process {file_path} Successful!")
    except Exception as e:
        print(f"Process {file_path} Failed with error: {e}")

def worker(gpu_id, image_files, input_dir, output_dir, config, args):
    # 初始化每个进程的pipeline
    pipeline = create_pipeline(pipeline="layout_parsing", device=f"gpu:{gpu_id}")
    for file_path in image_files:
        process_image(file_path, input_dir, output_dir, pipeline, config, args)

def process_images(input_dir, output_dir, num_gpus, processes_per_gpu, config, args):
    # 收集所有图片文件路径
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    print(f"Total images found: {len(image_files)}")

    # 将图片文件分配到每个GPU和进程
    chunk_size = len(image_files) // (num_gpus * processes_per_gpu)
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    # 创建多进程
    processes = []
    for gpu_id in range(num_gpus):
        for _ in range(processes_per_gpu):
            if chunks:
                chunk = chunks.pop(0)
                p = multiprocessing.Process(target=worker, args=(gpu_id, chunk, input_dir, output_dir, config, args))
                processes.append(p)
                p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, )
    parser.add_argument('--output_dir', type=str, required=True, )
    parser.add_argument('--num_gpus', type=int, default=1, )
    parser.add_argument('--processes_per_gpu', type=int, default=1, )
    parser.add_argument('--access_token', type=str, required=True, )
    parser.add_argument('--ernie_model_name', type=str, default='ernie-4.0', )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    num_gpus = args.num_gpus
    processes_per_gpu = args.processes_per_gpu
    config = initialize_model(access_token=args.access_token)
    
    process_images(input_dir, output_dir, num_gpus, processes_per_gpu, config, args)
