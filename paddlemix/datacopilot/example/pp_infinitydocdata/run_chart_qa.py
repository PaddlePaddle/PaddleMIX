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
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import time
import random
import requests # type: ignore
from pathlib import Path


import shutil
import hashlib
import base64

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
def read_csv_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_types_from_template(chart_type):
    if chart_type== 'histogram' or chart_type == 'bar_chart' or chart_type == 'stacked_bar_chart':
        return "数据检索,数据检索,找极值,确定范围,做比较"
    elif chart_type == '100_stacked_bar_chart' or chart_type == 'pie_chart':
        return "数据检索,数据检索,找极值,做比较"
    elif chart_type == 'area_chart' or chart_type == 'line_chart' or chart_type == 'stacked_area_chart':
        return "数据检索,数据检索,找极值,确定范围,寻找相关性/趋势,做比较"
    elif chart_type == 'scatterplot':
        return "数据检索,数据检索,找极值,确定范围,描述分布,发现异常,发现聚类,寻找相关性/趋势,做比较"
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")




template = '''
```json
[
    {
        "from": "human",
        "type_of_question": "数据检索",
        "value": "xxx"
    },
    {
        "from": "gpt",
        "value": "xxx"
    },
    {
        "from": "human",
        "type_of_question": "寻找极值",
        "value": "xxx"
    },
    {
        "from": "gpt",
        "value": "xxx"
    }
]
```
'''

def read_file_content(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            return json.load(file)
        else:
            return file.read()


def generate_qa_pairs(args):
    code_data, chart_name, save_dir, table_data, task_types,chart_type, config, model_name = args
    question = f'''你是一个熟悉数据可视化和{chart_type}的智能AI。
    以下是{chart_type}图表的matplotlib代码：{code_data}和对应的表格数据：{table_data}。
    请想象你正在查看代码生成的图像，而不是代码本身。
    请根据图表内容生成不同任务类型的问题。
    任务类型是{task_types}。
    记住在你的回答中，只有图表的图像是给定的，你的回答基于该图像, 表格数据是图像中相关数值的真实值，因此请确保答案正确。
    问题的值和标签是你问题的真实依据，所以请确保答案正确。
    避免在字符串中使用无效的转义字符。
    使用大致的颜色名称而不是十六进制颜色。
    如果图表中有单位、%等信息，请在回答中确保单位和%等信息的完整性。
    另外，我希望将你的输出保存为一个json文件，所以希望你可以像{template}一样组织你的答案，并确保json中的human和gpt的值都是中文。
    '''
    try:
        assistant_reply = predict(prompts=question, model_name=model_name, config=config)
    except Exception as e:
        print(f"Error processing {chart_name}: {e}")
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
        json_path = f'{save_dir}/{os.path.splitext(chart_name)[0]}.json'  # 强制保存为 .json 文件
        with open(json_path, 'w') as file:
            json.dump(qa_pairs, file, indent=4,ensure_ascii=False)
        print(f"Save qa_pairs to {json_path}")
    except Exception as e:
        print(f"Error processing {chart_name}: {e}")



def main(chart_dir, ori_chart_dir, table_dir, max_workers, config, model_name):
    # chart_type_list = ['100_stacked_bar_chart','area_chart','bar_chart','histogram','line_chart','pie_chart','scatterplot','stacked_area_chart','stacked_bar_chart']
    chart_type_list=['100_stacked_bar_chart']
    for chart_type in chart_type_list:
        chart_dir=os.path.join(ori_chart_dir,chart_type)

        save_dir=chart_dir
        table_dir = chart_dir
        print(f'processing {chart_dir}...')
        task_types = extract_types_from_template(chart_type)
        chart_list = [f for f in os.listdir(chart_dir) 
                    if f.endswith(('.py'))  
                    and os.path.exists(os.path.join(chart_dir, f"{os.path.splitext(f)[0]}.png")) 
                    and os.path.exists(os.path.join(table_dir, f"{os.path.splitext(f)[0]}.csv")) ]
        
        args_list = [
            (read_file_content(os.path.join(chart_dir, chart_name)), chart_name, save_dir, read_csv_content(os.path.join(table_dir, f"{os.path.splitext(chart_name)[0]}.csv")), task_types, chart_type, config, model_name)
            for chart_name in chart_list if not os.path.exists(os.path.join(save_dir, f"{os.path.splitext(chart_name)[0]}.json"))
        ]

        len_data = len(args_list)
        if len_data==0:
            print('all charts have been processed {}'.format(chart_dir))
            continue
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            st_time = time.time()
            futures = [executor.submit(generate_qa_pairs, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(args_list), desc='Processing'):
                future.result()  
            cost_time = time.time() - st_time
            print(f'len_data: {len_data}')
            print(f'平均每条数据生成时间：{cost_time / len_data:.2f}秒')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart_dir', type=str,
                        required=True)
    parser.add_argument('--save_dir', type=str,
                        required=True)
    parser.add_argument('--output_file', type=str,
                        default='chart_output.json')
    parser.add_argument('--max_workers', type=int,
                        default=10, help='Number of worker processes')
    parser.add_argument('--multiprocessing', type=bool, default=True)
    parser.add_argument('--access_token', type=str, required=True, )
    parser.add_argument('--ernie_model_name', type=str, default='ernie-4.0', )
    args = parser.parse_args()
    
    config = initialize_model(access_token=args.access_token)
    os.makedirs(args.save_dir, exist_ok=True)
    main(chart_dir=args.chart_dir, ori_chart_dir=args.chart_dir, table_dir=args.chart_dir, max_workers=args.max_workers, config=config, model_name=args.ernie_model_name)
