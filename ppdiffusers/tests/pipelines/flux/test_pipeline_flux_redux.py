import gc
import unittest

import numpy as np
import pytest
import paddle

from ppdiffusers import FluxPipeline, FluxPriorReduxPipeline
from ppdiffusers.utils import load_image
from ppdiffusers.utils.testing_utils import (
    numpy_cosine_similarity_distance,
    slow,
)
from paddlemix.models.llava.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from paddlemix.models.llava.multimodal_encoder.siglip_encoder import SigLipVisionModel

@slow
# @pytest.mark.big_gpu_with_torch_cuda
class FluxReduxSlowTests(unittest.TestCase):
    pipeline_class = FluxPriorReduxPipeline
    repo_id = "black-forest-labs/FLUX.1-Redux-dev"
    base_pipeline_class = FluxPipeline
    base_repo_id = "black-forest-labs/FLUX.1-dev"

    def setUp(self):
        super().setUp()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        init_image = load_image(
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png"
        )
        return {"image": init_image}

    def get_base_pipeline_inputs(self, seed=0):
        paddle.seed(seed=seed)

        return {
            "num_inference_steps": 2,
            "guidance_scale": 2.0,
            "output_type": "np",
        }

    def test_flux_redux_inference(self):
        # TODO: Currently only siglip_model can be loaded separately
        siglip_model = SigLipVisionModel.from_pretrained(self.repo_id + "/image_encoder")
        siglip_processor = SigLipImageProcessor()
        pipe_redux = self.pipeline_class.from_pretrained(self.repo_id, paddle_dtype=paddle.bfloat16,
            feature_extractor = siglip_processor,
            image_encoder = siglip_model
        )
        pipe_base = self.base_pipeline_class.from_pretrained(
            self.base_repo_id, paddle_dtype=paddle.float16, text_encoder=None, text_encoder_2=None
        )
        # pipe_base.enable_model_cpu_offload()

        inputs = self.get_inputs()
        base_pipeline_inputs = self.get_base_pipeline_inputs()

        redux_pipeline_output = pipe_redux(**inputs)
        image = pipe_base(**base_pipeline_inputs, **redux_pipeline_output).images[0]

        image_slice = image[0, :10, :10]
        expected_slice = np.array(
            [
            0.3803711,
            0.5175781,
            0.62109375,
            0.36035156,
            0.5058594,
            0.62353516,
            0.36914062,
            0.50439453,
            0.6123047,
            0.3630371,
            0.4987793,
            0.6147461,
            0.35839844,
            0.4951172,
            0.61035156,
            0.3486328,
            0.4897461,
            0.6015625,
            0.3449707,
            0.48901367,
            0.6015625,
            0.34106445,
            0.48291016,
            0.5966797,
            0.3461914,
            0.48339844,
            0.59716797,
            0.35058594,
            0.48535156,
            0.59716797,
        ],
            dtype=np.float32,
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4
