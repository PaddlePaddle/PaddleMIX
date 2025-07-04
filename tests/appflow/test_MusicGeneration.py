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


class MusicGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_text2music(self):

        task = Appflow(app="music_generation", models=["cvssp/audioldm"])
        prompt = "A classic cocktail lounge vibe with smooth jazz piano and a cool, relaxed atmosphere."
        negative_prompt = "low quality, average quality, muffled quality, noise interference, poor and low-grade quality, inaudible quality, low-fidelity quality"
        audio_length_in_s = 5
        num_inference_steps = 20

        result = task(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            generator=paddle.Generator().manual_seed(120),
        )["result"]

        self.assertIsNotNone(result)

    # def test_image2music(self):
    #     task1 = Appflow(app="music_generation", models=["miniGPT4/MiniGPT4-7B"])
    #     negative_prompt = 'low quality, average quality, muffled quality, noise interference, poor and low-grade quality, inaudible quality, low-fidelity quality'
    #     audio_length_in_s = 5
    #     num_inference_steps = 20
    #     output_path = "tmp.wav"
    #     minigpt4_text = 'describe the image, '
    #     image_pil = load_image(self.url)

    #     result = task1(image=image_pil, minigpt4_text=minigpt4_text )['result'].split('#')[0]
    #     paddle.device.cuda.empty_cache()

    #     self.assertIsNotNone(result)


if __name__ == "__main__":

    unittest.main()
