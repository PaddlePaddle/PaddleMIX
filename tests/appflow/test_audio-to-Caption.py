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
from paddlemix.appflow import Appflow
import paddle
from tests.testing_utils import _run_slow_test


class OpenSetDetSamAppSlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass


if __name__ == "__main__":

    def create_test(name, static_mode):
        def test_openset_det_sam(self):

            paddle.seed(1024)
            task = Appflow(app="audio2caption", models=["whisper", "THUDM/chatglm-6b"])
            audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zh.wav")

            prompt = (
                "描述这段话：{}."
            )
            result = task(audio=audio_file, prompt=prompt)['prompt']

            self.assertIsNotNone(result)
            self.assertIn('健康', result)

        setattr(OpenSetDetSamAppSlowTest, name, test_openset_det_sam)

    create_test(name="test_dygraph", static_mode=False)
    if _run_slow_test:
        create_test(name="test_static", static_mode=True)

    unittest.main()