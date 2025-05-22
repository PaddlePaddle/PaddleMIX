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

from utils.hparams import hparams


class BaseAugmentation:
    """
    Base class for data augmentation.
    All methods of this class should be thread-safe.
    1. *process_item*:
        Apply augmentation to one piece of data.
    """

    def __init__(self, data_dirs: list, augmentation_args: dict):
        self.raw_data_dirs = data_dirs
        self.augmentation_args = augmentation_args
        self.timestep = hparams["hop_size"] / hparams["audio_sample_rate"]

    def process_item(self, item: dict, **kwargs) -> dict:
        raise NotImplementedError()


def require_same_keys(func):
    def run(*args, **kwargs):
        item: dict = args[1]
        res: dict = func(*args, **kwargs)
        assert set(item.keys()) == set(
            res.keys()
        ), f"""Item keys mismatch after augmentation.
Before: {sorted(item.keys())}
After: {sorted(res.keys())}"""
        return res

    return run
