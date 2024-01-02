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
import tempfile
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from paddlemix.appflow import Appflow
import paddle
from tests.testing_utils import _run_slow_test


class AudioToCaptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass


if __name__ == "__main__":

    def create_test(name, static_mode):
        def test_audio2caption(self):

            paddle.seed(1024)
            task = Appflow(app="audio2caption", models=["conformer_u2pp_online_wenetspeech", "THUDM/chatglm-6b"])
            audio_file_url = "https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/appflow/test/test_audio/zh.wav"

            prompt = (
                "描述这段话：{}."
            )

            with tempfile.NamedTemporaryFile() as audio_file:
                audio_file.write(requests.get(audio_file_url).content)
                result = task(audio=audio_file.name, prompt=prompt)['prompt']

                self.assertIsNotNone(result)
                self.assertIn('健康', result)

        setattr(AudioToCaptionTest, name, test_audio2caption)

    create_test(name="test_dygraph", static_mode=False)
    if _run_slow_test:
        create_test(name="test_static", static_mode=True)

    unittest.main()
