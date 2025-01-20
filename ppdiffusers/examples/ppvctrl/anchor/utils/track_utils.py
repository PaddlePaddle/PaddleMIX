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

import numpy as np


def sample_points_from_masks(masks, num_points):
    """
    sample points from masks and return its absolute coordinates

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = tuple(masks.shape)
    points = []
    for i in range(n):
        indices = np.argwhere(masks[i] == 1)
        indices = indices[:, ::-1]
        if len(indices) == 0:
            points.append(np.array([]))
            continue
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)
    points = np.array(points, dtype=np.float32)
    return points
