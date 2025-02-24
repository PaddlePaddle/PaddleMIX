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

from typing import Any, Optional, Union

import paddle
from omegaconf import DictConfig, OmegaConf


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        paddle.distributed.broadcast(tensor=tensor, src=src)
        return tensor


def _distributed_available():
    return paddle.distributed.is_available() and paddle.distributed.is_initialized()


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    if "--local-rank" in cfg:
        del cfg["--local-rank"]
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
