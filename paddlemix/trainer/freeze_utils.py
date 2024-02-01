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

from fnmatch import fnmatch
from typing import List, Optional

import paddle
from paddlenlp.utils.log import logger


def freeze_params(
    model: paddle.nn.Layer, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None
) -> None:

    logger.info(f"Freeze parameters: {include} and exclude parameters: {exclude}")

    if not isinstance(model, paddle.nn.Layer):
        raise TypeError(f"model should be paddle.nn.Layer, but received {type(model)}")

    if include is None and exclude is None:
        raise ValueError("Either include or exclude should be provided.")

    return _exclude_freeze(model, include, exclude)


def _exclude_freeze(model: paddle.nn.Layer, include: List[str], exclude: List[str]) -> None:
    for name, param in model.named_parameters():
        if _match_name(name, include) and not _match_name(name, exclude):
            param.stop_gradient = True
        elif _match_name(name, exclude) and not _match_name(name, include):
            param.stop_gradient = False

    logger.info("Freeze parameters successfully.")
    return model


def _match_name(name: str, patterns: List[str]) -> bool:
    if patterns is None:
        return False

    for pattern in patterns:

        if fnmatch(name, pattern):
            return True

    return False
