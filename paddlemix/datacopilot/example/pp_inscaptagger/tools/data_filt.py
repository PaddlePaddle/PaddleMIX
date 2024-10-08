import json
import re
import os
import requests
from io import BytesIO
from PIL import Image
from datacopilot.core import MMDataset
import datacopilot.hub as hub
from functools import partial
import ast

tag_num = 3
def process(item, all_tags):
    try:
        tags = ast.literal_eval(item["tag"])['tags']
        tags = set(tags)
        
        tag_counts=len(tags)

        if tag_counts < tag_num and tags - all_tags == set():
            return None
        else:
            return item
    except:
        return item

if __name__ == '__main__':
    tag_most_ratio = 0.007
    all_tags = set()
    path = 'path/to/your/tag_file.json'
    tag_path = 'path/to/your/tag_file_tag_count.json'
    tag_count_list = MMDataset.from_json(tag_path)
    tag_num = len(tag_count_list)
    print(f'{path}数据集tags的种类总数为:',tag_num)

    tag_used_num = int(tag_num*tag_most_ratio)
    print(f'数量占比前{tag_most_ratio}的tags的种类总数为:',tag_used_num)
    for t,n in tag_count_list[:tag_used_num]:
        all_tags.add(t)
    print(f'使用的前{tag_most_ratio}%的tags:',all_tags)

    dataset = MMDataset.from_json(path)
    data_len = len(dataset)
    print('原始数据集长度:',data_len)
    func = partial(
        process, 
        all_tags=all_tags
    )
    dataset = dataset.map(func)
    newdataset = dataset.nonempty()
    out_data_len = len(newdataset)
    print('筛选后数据集数量:',out_data_len)
    print('筛选后数据集占原数据集比例: ', out_data_len/data_len)

    newdataset.export_json(path.replace('.json', f'_filter_{out_data_len}_tag.json'))