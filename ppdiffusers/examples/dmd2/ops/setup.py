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

import multiprocessing
import os
from site import getsitepackages

import paddle

paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include", "third_party"))
    paddle_includes.append(os.path.join(site_packages_path, "nvidia", "cudnn", "include"))


def get_gencode_flags(compiled_all=False):
    if not compiled_all:
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    else:
        return [
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_70,code=sm_70",
        ]


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


def run_single(func):
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def run_multi(func_list):
    processes = []
    for func in func_list:
        processes.append(multiprocessing.Process(target=func))
        processes.append(multiprocessing.Process(target=func))
        processes.append(multiprocessing.Process(target=func))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


cc_flag = get_gencode_flags(compiled_all=False)
cc = get_sm_version()


def setup_paddle_bwd_ops():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    sources = ["softmax_bwd/softmax_bwd.cc"]

    setup(
        name="paddle_bwd_ops",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=sources,
        ),
    )


if __name__ == "__main__":
    run_multi(
        [
            setup_paddle_bwd_ops,
        ],
    )
