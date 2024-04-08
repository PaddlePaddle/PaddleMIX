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

import unittest

import paddle
from paddle import nn

from ppdiffusers.models.activations import get_activation


class ActivationsTests(unittest.TestCase):
    def test_swish(self):
        act = get_activation("swish")

        self.assertIsInstance(act, nn.Silu)

        self.assertEqual(act(paddle.to_tensor(-100, dtype=paddle.float32)).item(), 0)
        self.assertNotEqual(act(paddle.to_tensor(-1, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(0, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(20, dtype=paddle.float32)).item(), 20)

    def test_silu(self):
        act = get_activation("silu")

        self.assertIsInstance(act, nn.Silu)

        self.assertEqual(act(paddle.to_tensor(-100, dtype=paddle.float32)).item(), 0)
        self.assertNotEqual(act(paddle.to_tensor(-1, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(0, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(20, dtype=paddle.float32)).item(), 20)

    def test_mish(self):
        act = get_activation("mish")

        self.assertIsInstance(act, nn.Mish)

        self.assertEqual(act(paddle.to_tensor(-200, dtype=paddle.float32)).item(), 0)
        self.assertNotEqual(act(paddle.to_tensor(-1, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(0, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(20, dtype=paddle.float32)).item(), 20)

    def test_gelu(self):
        act = get_activation("gelu")

        self.assertIsInstance(act, nn.GELU)

        self.assertEqual(act(paddle.to_tensor(-100, dtype=paddle.float32)).item(), 0)
        self.assertNotEqual(act(paddle.to_tensor(-1, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(0, dtype=paddle.float32)).item(), 0)
        self.assertEqual(act(paddle.to_tensor(20, dtype=paddle.float32)).item(), 20)
