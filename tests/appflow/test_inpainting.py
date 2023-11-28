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
from tests.testing_utils import slow


@slow
class GroundedSAMInpaintingAppSlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
        cls.expected_image = load_numpy(
            "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/inpainting-bus.npy"
        )
        cls.task = Appflow(
            app="inpainting",
            models=[
                "GroundingDino/groundingdino-swint-ogc",
                "Sam/SamVitH-1024",
                "stabilityai/stable-diffusion-2-inpainting",
            ],
        )

    def test_dygraph(self):
        paddle.seed(1024)
        image_pil = load_image(self.url)
        result = self.task(image=image_pil, prompt="bus", inpaint_prompt="A school bus parked on the roadside")

        boxes = np.array([112, 118, 513, 382])
        avg_diff = np.abs(result["boxes"][0] - boxes).mean()
        assert avg_diff < 5, f"Error bbox deviates {avg_diff} pixels on average"

        avg_diff = np.abs(np.array(result["result"]) - self.expected_image).mean()
        assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"


@slow
class GroundedSAMChatglmAppSlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
        cls.expected_image = load_numpy(
            "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/chat-inpainting-bus.npy"
        )
        cls.task = Appflow(
            app="inpainting",
            models=[
                "THUDM/chatglm-6b",
                "GroundingDino/groundingdino-swint-ogc",
                "Sam/SamVitH-1024",
                "stabilityai/stable-diffusion-2-inpainting",
            ],
        )

    def test_dygraph(self):
        paddle.seed(1024)
        image_pil = load_image(self.url)
        inpaint_prompt = "bus is changed to A school bus parked on the roadside"
        prompt = (
            "Given caption,extract the main object to be replaced and marked it as 'main_object',"
            + "Extract the remaining part as 'other prompt', "
            + "Return main_object, other prompt in English"
            + "Given caption: {}.".format(inpaint_prompt)
        )
        result = self.task(image=image_pil, prompt=prompt)

        boxes = np.array([112, 118, 513, 382])
        avg_diff = np.abs(result["boxes"][0] - boxes).mean()
        assert avg_diff < 5, f"Error bbox deviates {avg_diff} pixels on average"

        avg_diff = np.abs(np.array(result["result"]) - self.expected_image).mean()
        assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"


@slow
class TextGuideInpaintingAppSlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        cls.mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
        cls.expected_image = load_numpy(
            "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/inpainting-cat.npy"
        )
        cls.task = Appflow(app="inpainting", models=["stabilityai/stable-diffusion-2-inpainting"])

    def test_dygraph(self):

        image = load_image(self.url)
        mask_image = load_image(self.mask_url)
        paddle.seed(1024)
        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        result = self.task(inpaint_prompt=prompt, image=image, seg_masks=mask_image)

        avg_diff = np.abs(np.array(result["result"]) - self.expected_image).mean()
        assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"


if __name__ == "__main__":

    unittest.main()
