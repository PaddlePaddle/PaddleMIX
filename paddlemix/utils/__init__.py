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

import contextlib
from unittest.case import _IsInstanceClassInfo
import paddle
import os

@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


def paddlemix_load(path,model,training_args, map_location="cpu"):
    assert map_location in ["cpu", "gpu", "xpu", "npu", "numpy", "np"]
    return load_model(training_args,model,training_args,ckpt_dir=path,map_location="cpu")
    
        
def load_model(args, model, optimizer=None, ckpt_dir="",map_location="cpu"):
    """
    load the saved checkpoint file and update the state dicts of model and optimizer.
    """
    from  paddlemix.examples.blip2.utils import get_weights_path_from_url
    from paddlemix.utils.downloader import is_url
    if ckpt_dir is None:
        return

    if not os.path.exists(ckpt_dir):
        ValueError("Cannot find pretrained model path: {}".format(ckpt_dir))

    if os.path.isfile(ckpt_dir):
        path = ckpt_dir
    elif is_url(ckpt_dir):
        path = get_weights_path_from_url(ckpt_dir)
    else:
        assert os.path.exists(ckpt_dir), f"{ckpt_dir} not exist"

    ckpt_dir = path

    print("Try to load a whole checkpoint from %s " % ckpt_dir)
    embedding_list = []
    collinear_list = []
    rowlinear_list = []
    skip_list = ["visual_encoder.patch_embed.proj.weight", "visual_encoder.patch_embed.proj.bias"]

    col_list = []
    row_list = []
    emb_list = []

    mp_rank = args.mp_rank
    mp_size = args.tensor_parallel_degree

    def renamebias(model_dict, whole_key):
        if "q_bias" in whole_key:
            key = whole_key.replace("q_bias", "q_proj.bias")
        elif "v_bias" in whole_key:
            key = whole_key.replace("v_bias", "v_proj.bias")
        model_dict[key] = model_dict[whole_key]
        del model_dict[whole_key]
        return model_dict

    def col_split_modeldict(model_dict):
        if len(model_dict.shape) == 2:
            subbatch = model_dict.shape[1] // mp_size
            return model_dict[:, mp_rank * subbatch : (mp_rank + 1) * subbatch]
        elif len(model_dict.shape) == 1:
            subbatch = model_dict.shape[0] // mp_size
            return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

    def row_split_modeldict(model_dict):
        if len(model_dict.shape) == 2:
            subbatch = model_dict.shape[0] // mp_size
            return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]
        else:
            return model_dict

    def emb_split_modeldict(model_dict):
        subbatch = model_dict.shape[0] // mp_size
        return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

    if map_location in ["numpy", "np"]:
        model_dict=paddle.load(path, return_numpy=True)
    else:
        with device_guard(map_location):
            model_dict=paddle.load(path)
    from paddle.distributed import fleet
    for name,p in model.named_sublayers():
        if isinstance(name, fleet.meta_parallel.ColumnParallelLinear):
            collinear_list.append(name)
        if isinstance(name, fleet.meta_parallel.RowParallelLinear):  
            rowlinear_list.append(name)
    all_list = collinear_list + rowlinear_list + embedding_list

    for whole_key in model_dict.keys():
        if "." not in whole_key:
            continue

        key = whole_key.split(".")[-2]
        if whole_key in skip_list:
            continue
        if key in all_list:
            if key in collinear_list:
                col_list.append((key, model_dict[whole_key].shape))
                model_dict[whole_key] = col_split_modeldict(model_dict[whole_key])
            elif key in rowlinear_list:
                row_list.append((key, model_dict[whole_key].shape))
                model_dict[whole_key] = row_split_modeldict(model_dict[whole_key])
            else:
                emb_list.append((key, model_dict[whole_key].shape))
                model_dict[whole_key] = emb_split_modeldict(model_dict[whole_key])

    return model_dict
