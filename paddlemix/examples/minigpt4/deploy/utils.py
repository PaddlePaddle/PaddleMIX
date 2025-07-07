# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import glob
import os
import struct

import numpy as np


def deserialize_from_file(fp):
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


def load_real_time_tokens():
    tokens = []
    files = glob.glob(os.path.join("./real_time_save.*"))
    for j in range(1, len(files) + 1):
        filename = "./real_time_save.temp_ids_rank_0_step_{}".format(j)
        if not os.path.exists(filename):
            break
        fp = open(filename, "rb+")
        fp.read(1)
        data_list = deserialize_from_file(fp)
        fp.close()
        tokens.append(np.array(data_list).reshape(-1, 1))
    os.system("rm -f ./real_time_save.temp_ids_rank_*")
    tokens = np.concatenate(tokens, axis=1)
    return tokens
