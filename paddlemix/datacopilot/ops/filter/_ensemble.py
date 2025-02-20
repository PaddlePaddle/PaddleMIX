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

from collections.abc import Sequence

from paddlemix.datacopilot.core import register
from paddlemix.datacopilot.core.schema import T

from ._tagger import Tagger


@register(force=True)
def ensemble(
    item: T,
    taggers: Sequence[Tagger | str],
    weights: Sequence[float],
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> bool:
    """
    Filter item with model ensemble.

    Args:
        item(Dict[str, Any]): Input dict with key `image`.
        taggers(Sequence[Tagger | str]): Taggers or keys.
        weights(Sequence[float]): Tag scores.
        min_score(float): The min score the image should reach. Default is 0.
        max_score(float): The max score the image can be. Default is 1.
    Returns:
        bool, return `True` if the score is within `[min_score, max_score]`.
    """
    assert 0 <= min_score < max_score <= 1
    assert len(taggers) == len(weights)

    score = 0
    for i in range(len(taggers)):
        tagger = taggers[i]
        if isinstance(tagger, Tagger):
            key = tagger.key()
        else:
            key = tagger

        if key in item:
            score += item[key] * weights[i]
        else:
            # if key not in item, just keep this item
            return 1.0

    if min_score <= score <= max_score:
        return True

    return False
