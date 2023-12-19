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

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image, load_numpy


class TextGuidedImageUpscalingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
    
    
    def test_image_upscaling(self):

        low_res_img = load_image(self.url).resize((128, 128))

        prompt = "a white cat"

        app = Appflow(app='image2image_text_guided_upscaling',models=['stabilityai/stable-diffusion-x4-upscaler'])
        image = app(prompt=prompt,image=low_res_img)['result']

        self.assertIsNotNone(image)


if __name__ == "__main__":

    unittest.main()
