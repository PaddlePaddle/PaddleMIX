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
import paddle

class TextGuidedImageInpaintingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        cls.mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

    def test_image_inpainting(self):
        image = load_image(self.img_url)
        mask_image = load_image(self.mask_url)
        paddle.seed(1024)

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        app = Appflow(app='inpainting',models=['stabilityai/stable-diffusion-2-inpainting'])
        image = app(inpaint_prompt=prompt,image=image,seg_masks=mask_image)['result']

        self.assertIsNotNone(image)


if __name__ == "__main__":

    unittest.main()
