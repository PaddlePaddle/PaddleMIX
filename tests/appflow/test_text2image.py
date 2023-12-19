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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import paddle
from paddlemix.appflow import Appflow


class Text2ImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass


if __name__ == "__main__":

    def create_test(name, static_mode):
        def test_openset_det_sam(self):
            paddle.seed(1024)
            task = Appflow(app="text2image_generation",
                        models=["stabilityai/stable-diffusion-v1-5"]
                        )
            prompt = "a photo of an astronaut riding a horse on mars."
            result = task(prompt=prompt)['result']

            self.assertIsNotNone(result)
            # todo: 增加结果对比
            

        setattr(Text2ImageTest, name, test_openset_det_sam)

    create_test(name="test_dygraph", static_mode=False)

    unittest.main()
