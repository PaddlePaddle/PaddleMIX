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



import json
import os
import re
import sys
from pdb import line_prefix

import numpy as np
from numpy import mean, var

class TimeAnalyzer(object):
    def __init__(self, filename, keyword=None, loss_keyword=None):
        if filename is None:
            raise Exception("Please specify the filename!")

        if keyword is None:
            raise Exception("Please specify the keyword!")

        self.filename = filename
        self.keyword = keyword
        self.loss_keyword = loss_keyword

    def get_ips(self):
        ips_list = []
        loss_list = []
        loss_value = None
        with open(self.filename, "r") as f_object:
            lines = f_object.read().splitlines()
            for line in lines:
                if self.keyword not in line:
                    continue
                try:
                    # result = None

                    # # Distill the string from a line.
                    # line = line.strip()
                    # line_words = line.split()
                    # for i in range(len(line_words) - 1):
                    #     if line_words[i] == self.keyword:
                    #         result = float(line_words[i + 1].replace(',', ''))
                    #         ips_list.append(result)
                    #     if line_words[i] == self.loss_keyword:
                    #         # 剔除掉该值后面的逗号并保留5位小数点
                    #         loss_value = line_words[i + 1].replace(',', '')  
                    #         # 保留5位小数
                    #         # loss_value = float("{:.5f}".format(float(loss_str_without_comma)))
                            
                    # # Distil the result from the picked string.

                    # 提取 ips
                    ips_match = re.search(r'(\d+\.\d+)it/s', line)
                    if ips_match:
                        ips = float(ips_match.group(1))
                        ips_list.append(ips)

                    # 提取 loss
                    loss_match = re.search(r'loss=(\d+\.\d+)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        loss_list.append(loss)
                        loss_value = loss

                except Exception as exc:
                    print("line is: {}; failed".format(line))
                    print("Exception: {}".format(exc))
        if loss_value is None:
            loss_value = -1
        def ewma(data, alpha):
            smoothed_data = []
            for i, value in enumerate(data):
                if i == 0:
                    smoothed_data.append(value)
                else:
                    smoothed_value = alpha * value + (1 - alpha) * smoothed_data[-1]
                    smoothed_data.append(smoothed_value)
            return smoothed_data
        smoothed_loss = ewma(loss_list, 0.9)[-1]
        return mean(ips_list[4:]), loss_value, smoothed_loss


def analyze(model_item, log_file, res_log_file, device_num, bs, fp_item):

    analyzer = TimeAnalyzer(log_file, 'Steps:', None)
    ips, convergence_value, smoothed_value = analyzer.get_ips()
    ips = round(ips, 3)
    # with open(str(log_file), "r", encoding="utf8") as f:
    #     data = f.readlines()
    # ips_lines = []
    # for eachline in data:
    #     if "train_samples_per_second:" in eachline:
    #         ips = float(eachline.split("train_samples_per_second: ")[1].split()[0].replace(',', ''))
    #         print("----ips: ", ips)
    #         ips_lines.append(ips)
    # print("----ips_lines: ", ips_lines)
    # ips = np.round(np.mean(ips_lines), 3)
    ngpus = int(re.findall("\d+", device_num)[-1])
    batch_size = int(re.findall("\d+", str(bs))[-1])
    print("----ips: ", ips, "ngpus", ngpus, "batch_size", batch_size)
    ips *= batch_size
    ips *= ngpus
    run_mode = "DP"

    model_name = model_item + "_" + "bs" + str(bs) + "_" + fp_item + "_" + run_mode
    info = {
        "model_branch": os.getenv("model_branch"),
        "model_commit": os.getenv("model_commit"),
        "model_name": model_name,
        "batch_size": bs,
        "fp_item": fp_item,
        "run_mode": run_mode,
        "convergence_value": convergence_value,
        "smoothed_value": smoothed_value,
        "convergence_key": "",
        "ips": ips,
        "speed_unit": "sample/sec",
        "device_num": device_num,
        "model_run_time": os.getenv("model_run_time"),
        "frame_commit": "",
        "frame_version": os.getenv("frame_version"),
    }
    json_info = json.dumps(info)
    print(json_info)
    with open(res_log_file, "w") as of:
        of.write(json_info)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()
    

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    bs = int(sys.argv[5])
    fp_item = sys.argv[6]

    analyze(model_item, log_file, res_log_file, device_num, bs, fp_item)