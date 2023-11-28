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
import paddle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image, load_numpy
from tests.testing_utils import _run_slow_test


class OpenSetDetSamAppSlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        cls.expected_image = load_numpy(
            "https://bj.bcebos.com/v1/paddlenlp/models/community/Sam/SamVitH-1024/overture-creations-mask.npy"
        )


if __name__ == "__main__":

    def create_test(name, static_mode):
        def test_openset_det_sam(self):
            paddle.seed(1024)
            self.task = Appflow(
                app="openset_det_sam",
                models=["GroundingDino/groundingdino-swint-ogc", "Sam/SamVitH-1024"],
                static_mode=static_mode,
            )
            prompt = "dog"
            image_pil = load_image(self.url)
            result = self.task(image=image_pil, prompt=prompt)

            boxes = np.array([174, 115, 311, 465])
            avg_diff = np.abs(result["boxes"][0] - boxes).mean()
            assert avg_diff < 5, f"Error bbox deviates {avg_diff} pixels on average"

            avg_diff = np.abs(
                result["seg_masks"][0].cpu().numpy().astype(int) - self.expected_image.astype(int)
            ).mean()
            assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"

        setattr(OpenSetDetSamAppSlowTest, name, test_openset_det_sam)

    create_test(name="test_static", static_mode=False)

    if _run_slow_test:
        create_test(name="test_static", static_mode=True)

    unittest.main()
