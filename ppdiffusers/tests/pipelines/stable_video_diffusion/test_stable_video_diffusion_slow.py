# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import gc
import unittest

import numpy as np
import paddle

from ppdiffusers import StableVideoDiffusionPipeline
from ppdiffusers.utils import load_image
from ppdiffusers.utils.testing_utils import (
    numpy_cosine_similarity_distance,
    require_paddle_gpu,
    slow,
)


@slow
@require_paddle_gpu
class StableVideoDiffusionPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_sd_video(self):
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            variant="fp16",
            paddle_dtype=paddle.float16,
        )
        pipe.set_progress_bar_config(disable=None)
        image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/cat_6.png"
        )

        generator = paddle.Generator().manual_seed(0)
        num_frames = 3

        output = pipe(
            image=image,
            num_frames=num_frames,
            generator=generator,
            num_inference_steps=25,
            output_type="np",
        )

        image = output.frames[0]
        assert image.shape == (num_frames, 576, 1024, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.8592, 0.8645, 0.8499, 0.8722, 0.8769, 0.8421, 0.8557, 0.8528, 0.8285])
        assert numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice.flatten()) < 1e-3
