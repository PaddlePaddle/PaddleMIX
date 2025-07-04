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


class AutoLabelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        cls.expected_image = load_numpy(
            "https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/appflow/test/test_autolabel/test_autolabel_mask.npy"
        )
        cls.expected_labels = ["dog", "bench", "field"]
        cls.expected_boxes = np.array([[174, 116, 312, 467], [32, 223, 464, 491], [0, 141, 511, 511]])

    def test_autolable(self):
        self.task = Appflow(
            app="auto_label",
            models=["paddlemix/blip2-caption-opt2.7b", "GroundingDino/groundingdino-swint-ogc", "Sam/SamVitH-1024"],
        )
        image_pil = load_image(self.url)
        blip2_prompt = "describe the image"
        result = self.task(image=image_pil, blip2_prompt=blip2_prompt)

        # check labels
        # res_labels like:  ['dog(0.72)', 'bench(0.63)', 'field(0.49)']
        res_labels_name = [label.split("(")[0] for label in result["labels"]]
        res_label_expected_label_map = []
        for expected_l in self.expected_labels:
            self.assertIn(expected_l, res_labels_name)

        for label in res_labels_name:
            res_label_expected_label_map.append(self.expected_labels.index(label))
        assert len(result["boxes"]) == len(self.expected_boxes), "Error box num"
        for i, box in enumerate(result["boxes"]):
            avg_diff = np.abs(self.expected_boxes[res_label_expected_label_map[i]] - box).mean()
            assert avg_diff < 5, f"Error bbox deviates {avg_diff} pixels on average"

        avg_diff = np.abs(result["seg_masks"][0].cpu().numpy().astype(int) - self.expected_image.astype(int)).mean()
        assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"


if __name__ == "__main__":

    unittest.main()
