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

import paddle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image


class Image2ImageTextGuidedGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
        cls.expected_image = "https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/appflow/test/test_Image2ImageTextGuidedGeneration/image2image_text_guided_generation.png"

    def test_image_generation(self):
        image = load_image(self.url).resize((512, 768))
        prompt = "a red car in the sun"

        paddle.seed(42)
        prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

        app = Appflow(app="image2image_text_guided_generation", models=["admruul/anything-v3.0"])
        image = app(prompt=prompt, negative_prompt=negative_prompt, image=image)["result"]

        self.assertIsNotNone(image)
        # 增加结果对比
        expect_img = load_image(self.expected_image)

        size = (512, 512)
        image1 = image.resize(size)
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
