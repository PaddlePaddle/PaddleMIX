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
from __future__ import annotations

from typing import Any

import numpy as np
from brisque import BRISQUE
from PIL import Image

from ...core import register, T

from ._tagger import Tagger

_brisque = BRISQUE(url=False)


def _score(model, img_path):
    img = Image.open(img_path)
    if img.mode not in ["RGB", "BGR"]:
        img = img.convert("RGB")

    img = np.asarray(img)
    score = model.score(img=img)

    # clip score to [0, 100]
    score = max(0, score)
    score = min(100, score)

    # convert BRISQUE score [0, 100] to [0, 1], and higher is better
    score = (100 - score) / 100
    return float(score)


class BRISQUETagger(Tagger):
    def __init__(self, model: Any = None):
        self._model = model or _brisque

    def key(self) -> str:
        return "__tag_brisque"

    def score(self, item: T, key: str = "image") -> float:
        if key not in item:
            return 1.0

        return _score(self._model, item[key])


tag_brisque = BRISQUETagger()


@register(force=True)
def iqa_brisque(
    item: T,
    min_score: float = 0.0,
    max_score: float = 1.0,
    key: str = "image",
) -> bool:
    """
    Filter image with BRISQUE `<https://ieeexplore.ieee.org/document/6272356>`_.
    The input item is kept with the score within `[min_score, max_score]`.
    The higher the score, the better the assessment of the image.

    Args:
        item(Dict[str, Any]): Input dict with key `image`.
        min_score(float): The min score the image should reach. Default is 0.
        max_score(float): The max score the image can be. Default is 1.
        key(str): The `image` key in `item`.
    Returns:
        bool, return `True` if the score is within `[min_score, max_score]`.

    """
    assert 0 <= min_score < max_score <= 1

    if key not in item:
        return True

    score = _score(_brisque, item[key])
    if min_score <= score <= max_score:
        return True

    return False
