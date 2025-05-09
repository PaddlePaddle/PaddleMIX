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

import numpy as np
import paddle.nn as nn

from ..utils import logging
from ..utils.import_utils import is_torch_available

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch


def convert_pytorch_state_dict_to_paddle(self: nn.Layer, pt_state_dict, sub_layer=None):
    # Step 1: Find Linear layer which need transpose weight
    linear_need_transpose = []
    for k, v in self.named_sublayers(include_self=True):
        if isinstance(v, nn.Linear):
            if sub_layer is not None and sub_layer not in k:
                continue
            linear_need_transpose.append(k + ".weight")

    ignore_keys = ["position_ids", ".num_batches_tracked"]
    ptname2pdname = {
        # torch.nn.BatchNorm2d -> paddle.nn.BatchNorm2D
        ".running_var": "._variance",
        ".running_mean": "._mean",
    }
    # Need to change some parameters name to match paddle names
    keys = list(pt_state_dict.keys())
    for pt_key in keys:
        pt_tensor = pt_state_dict.pop(pt_key)
        # only convert sub_layer state dict
        if sub_layer is not None and sub_layer not in pt_key:
            continue
        # (0) ignore_keys
        if any(i in pt_key for i in ignore_keys):
            continue
        # (1) transpose linear
        if pt_key in linear_need_transpose and pt_tensor.ndim == 2:
            pt_tensor = pt_tensor.T
        # (2) 0d tensor -> 1d tensor
        # if pt_tensor.ndim == 0:
        # pt_tensor = pt_tensor.reshape((1,))
        # (3) name mapping
        for old_key, new_key in ptname2pdname.items():
            pt_key = pt_key.replace(old_key, new_key)

        pt_state_dict[pt_key] = pt_tensor
    return pt_state_dict


def convert_paddle_state_dict_to_pytorch(self: nn.Layer, pd_state_dict):
    # Step 2: Find Linear layer which need transpose weight
    linear_need_transpose = []
    for k, v in self.named_sublayers(include_self=True):
        if isinstance(v, nn.Linear):
            linear_need_transpose.append(k + ".weight")

    ignore_keys = ["position_ids"]
    ptname2pdname = {
        # torch.nn.BatchNorm2d -> paddle.nn.BatchNorm2D
        ".running_var": "._variance",
        ".running_mean": "._mean",
    }
    keys = list(pd_state_dict.keys())
    detect_bfloat16 = False
    for pd_key in keys:
        pd_tensor = pd_state_dict.pop(pd_key)
        # (0) ignore_keys
        if any(i in pd_key for i in ignore_keys):
            continue
        # (1) transpose linear
        if pd_key in linear_need_transpose and pd_tensor.ndim == 2:
            pd_tensor = pd_tensor.T
        # TODO maybe not true
        # (2) 1d tensor -> 0d tensor
        if pd_tensor.ndim == 1:
            pd_tensor = pd_tensor.squeeze()
        # (3) name mapping
        for old_key, new_key in ptname2pdname.items():
            pd_key = pd_key.replace(new_key, old_key)

        pd_tensor = np.ascontiguousarray(pd_tensor)

        if is_torch_available():
            if pd_tensor.dtype in ["uint16", np.uint16]:
                pd_tensor = pd_tensor.astype(np.float32)
                pd_state_dict[pd_key] = torch.from_numpy(pd_tensor).to(paddle.bfloat16
)
            else:
                pd_state_dict[pd_key] = torch.from_numpy(pd_tensor)
        else:
            if pd_tensor.dtype in ["uint16", np.uint16]:
                pd_tensor = pd_tensor.astype(np.float16)
                detect_bfloat16 = True
            pd_state_dict[pd_key] = pd_tensor

    if detect_bfloat16:
        logger.warning(
            "PyTorch is not installed, so we cannot save as `bfloat16` tensor. "
            "To ensure the model can still be loaded, we will save it as `float16` tensor instead. "
            "Please note that this may affect the precision of the saved model."
        )
    return pd_state_dict
