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

import os
import sys
import unittest

from paddlemix.processors.image_utils import load_image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import paddle

from paddlemix.appflow import Appflow


class Text2ImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expect_img_url = "https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/appflow/test/test_text2image/riding_horse.png"

    def test_text2image(self):
        paddle.seed(1024)
        task = Appflow(app="text2image_generation", models=["stabilityai/stable-diffusion-v1-5"])
        prompt = "a photo of an astronaut riding a horse on mars."
        result = task(prompt=prompt)["result"]

        self.assertIsNotNone(result)
        # 增加结果对比
        expect_img = load_image(self.expect_img_url)

        size = (768, 768)
        image1 = result.resize(size)
        image2 = expect_img.resize(size)

        # 获取图像数据
        data1 = list(image1.getdata())
        data2 = list(image2.getdata())

        # 计算每个像素点的差值，并求平均值
        diff_sum = 0.0
        for i in range(len(data1)):
            diff_sum += sum(abs(c - d) for c, d in zip(data1[i], data2[i]))

        average_diff = diff_sum / len(data1)

        self.assertLessEqual(average_diff, 5)


if __name__ == "__main__":

    unittest.main()
