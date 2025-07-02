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

import paddle


class DummyCondStage:
    def __init__(self, conditional_key):
        self.conditional_key = conditional_key
        self.train = None

    def eval(self):
        return self

    @staticmethod
    def encode(c: paddle.Tensor):
        return c, None, (None, None, c)

    @staticmethod
    def decode(c: paddle.Tensor):
        return c

    @staticmethod
    def to_rgb(c: paddle.Tensor):
        return c
