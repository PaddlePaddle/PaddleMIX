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

import ast
import glob
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from paddlemix.datacopilot.core import MMDataset


def merge_json_files(folder_path):
    newdataset = MMDataset()
    pathes = sorted(glob.glob(f"{folder_path}/*.json"))
    file_count = len(pathes)
    for path in sorted(glob.glob(f"{folder_path}/*.json")):
        newdataset += MMDataset.from_json(path)
    output_file = f"merged_{file_count}.json"
    newdataset.export_json(output_file)
    return output_file


def all_tag_count(data_json):
    data = json.load(open(data_json, encoding="utf-8"))
    tag_counts = {}
    n = 0
    for item in data:
        try:
            tags = ast.literal_eval(item["tag"])["tags"]
            for tag in list(set(tags)):
                # 如果tag中包含逗号，则分割tag
                if "," in tag:
                    # 使用split()方法按照逗号分割字符串
                    split_strings = tag.split(",")
                    # 去除每个字符串两端的空格
                    tags = [s.strip() for s in split_strings]
                    for tag in tags:
                        if tag in tag_counts:
                            tag_counts[tag] += 1
                        else:
                            tag_counts[tag] = 1

                if tag in tag_counts:
                    tag_counts[tag] += 1
                else:
                    tag_counts[tag] = 1
        except:
            n += 1
    print("无效tag的数据数量：", n)
    print("数据集总量：", len(data))
    print("tag数量：", len(tag_counts))
    sorted_tag_counts = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
    output_file = data_json.replace(".json", "_tag_count.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_tag_counts, f, ensure_ascii=False, indent=4)
    return sorted_tag_counts


def one_data_tag_count(data_json):
    data = json.load(open(data_json, encoding="utf-8"))
    # 统计每条数据中tag的数量
    tag_counts = []
    for item in data:
        try:
            tags = ast.literal_eval(item["tag"])["tags"]
            tag_counts.append(len(tags))
        except:
            print(item["tag"])

    # 统计每个tag数量级别的数据条数
    tag_count_freq = Counter(tag_counts)
    # 按tag数量排序并计算累积数据覆盖数量
    sorted_tag_counts = sorted(tag_count_freq.items(), key=lambda x: x[0], reverse=True)
    # 将统计结果保存为字典
    tag_count_freq_dict = dict(sorted_tag_counts)
    # 将统计结果保存到JSON文件
    output_file = data_json.replace(".json", "_tag_count_statistics.json")
    with open(output_file, "w") as f:
        json.dump(tag_count_freq_dict, f, indent=4)

    cumulative_coverage = np.cumsum([count for _, count in sorted_tag_counts])
    # 找到覆盖90%和80%数据的最少tag数量
    total_data = len(tag_counts)
    cover_90_percent = next(
        tag
        for tag, cum_cov in zip([tag for tag, _ in sorted_tag_counts], cumulative_coverage)
        if cum_cov >= 0.8 * total_data
    )
    cover_80_percent = next(
        tag
        for tag, cum_cov in zip([tag for tag, _ in sorted_tag_counts], cumulative_coverage)
        if cum_cov >= 0.6 * total_data
    )
    print(f"可以覆盖90%数据的单条数据tag数量: {cover_90_percent}")
    print(f"可以覆盖80%数据的单条数据tag数量: {cover_80_percent}")


def tag_count_freq_plot(tag_count_file, tag_file, topn):
    data = json.load(open(tag_count_file, encoding="utf-8"))
    # 设置字体，使用您安装的中文字体名称
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]  # 例如：SimHei、Microsoft YaHei、WenQuanYi Zen Hei等
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    # 确保以UTF-8编码读取JSON文件
    data = json.load(open(tag_file, encoding="utf-8"))
    data = data[:topn]  # 绘制部分数据
    categories, values = zip(*data)
    plt.figure(figsize=(10, 30))
    plt.barh(categories, values, color="skyblue")
    plt.xlabel("数量")
    plt.title("类别分布")
    plt.yticks(fontsize=8)  # 调整字体大小以适应显示
    # 保存图形
    im_path = tag_file.replace(".json", "_plot.png")
    plt.savefig(im_path, bbox_inches="tight")
