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

from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.processors.groundingdino_processing import (
    GroudingDinoImageProcessor,
    GroudingDinoTextProcessor,
)
from tests.testing_utils import ai_studio_token, slow

repo_id = "aistudio/groundingdino-swint-ogc"
bos_model_name = "GroundingDino/groundingdino-swint-ogc"


class FromAiStudioTester:
    def __init__(self):
        self.bos_model_name = bos_model_name
        self.token = ai_studio_token
        self.repo_id = repo_id

    def prepare_from_bos(self):
        model = GroundingDinoModel.from_pretrained(self.bos_model_name)
        image_processor = GroudingDinoImageProcessor.from_pretrained(self.bos_model_name)
        text_processor = GroudingDinoTextProcessor.from_pretrained(self.bos_model_name)

        return model, image_processor, text_processor


class AIStudioUpTester(unittest.TestCase):
    def setUp(self):
        self.tester = FromAiStudioTester()
        self.model, self.image_processor, self.text_processor = self.tester.prepare_from_bos()

    @slow
    def test_model_up_aistusio(self):
        self.model.save_to_aistudio(
            repo_id=self.tester.repo_id,
            token=self.tester.token,
            private=True,
            license="Apache License 2.0",
            exist_ok=True,
            safe_serialization=True,
        )

    def test_processor_up_aistusio(self):
        self.image_processor.save_to_aistudio(
            repo_id=self.tester.repo_id,
            token=self.tester.token,
            private=True,
            license="Apache License 2.0",
            exist_ok=True,
        )
        self.text_processor.save_to_aistudio(
            repo_id=self.tester.repo_id,
            token=self.tester.token,
            private=True,
            license="Apache License 2.0",
            exist_ok=True,
        )


class AIStudioLoadTester(unittest.TestCase):
    def setUp(self):
        self.tester = FromAiStudioTester()

    @slow
    def test_model_load_aistusio(self):
        GroundingDinoModel.from_pretrained(self.tester.repo_id, from_aistudio=True)

    def test_processor_load_aistusio(self):
        GroudingDinoTextProcessor.from_pretrained(self.tester.repo_id, from_aistudio=True)

    def image_processor_load_aistusio(self):
        GroudingDinoImageProcessor.from_pretrained(self.tester.repo_id, from_aistudio=True)
