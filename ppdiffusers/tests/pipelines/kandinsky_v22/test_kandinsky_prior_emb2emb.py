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

import random
import unittest

import numpy as np
import paddle
from paddlenlp.transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from PIL import Image

from ppdiffusers import (
    KandinskyV22PriorEmb2EmbPipeline,
    PriorTransformer,
    UnCLIPScheduler,
)
from ppdiffusers.utils import floats_tensor
from ppdiffusers.utils.testing_utils import enable_full_determinism

from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism()


class KandinskyV22PriorEmb2EmbPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22PriorEmb2EmbPipeline
    params = ["prompt", "image"]
    batch_params = ["prompt", "image"]
    required_optional_params = [
        "num_images_per_prompt",
        "strength",
        "generator",
        "num_inference_steps",
        "latents",
        "negative_prompt",
        "guidance_scale",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def time_input_dim(self):
        return 32

    @property
    def block_out_channels_0(self):
        return self.time_input_dim

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

    @property
    def cross_attention_dim(self):
        return 100

    @property
    def dummy_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        paddle.seed(seed=0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModelWithProjection(config)

    @property
    def dummy_prior(self):
        paddle.seed(seed=0)
        model_kwargs = {
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "embedding_dim": self.text_embedder_hidden_size,
            "num_layers": 1,
        }
        model = PriorTransformer(**model_kwargs)
        out_84 = paddle.create_parameter(
            shape=paddle.ones(shape=model.clip_std.shape).shape,
            dtype=paddle.ones(shape=model.clip_std.shape).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=model.clip_std.shape)),
        )
        out_84.stop_gradient = not True
        model.clip_std = out_84
        return model

    @property
    def dummy_image_encoder(self):
        paddle.seed(seed=0)
        config = CLIPVisionConfig(
            hidden_size=self.text_embedder_hidden_size,
            image_size=224,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )
        model = CLIPVisionModelWithProjection(config)
        return model

    @property
    def dummy_image_processor(self):
        image_processor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )
        return image_processor

    def get_dummy_components(self):
        prior = self.dummy_prior
        image_encoder = self.dummy_image_encoder
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        image_processor = self.dummy_image_processor
        scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample=True,
            clip_sample_range=10.0,
        )
        components = {
            "prior": prior,
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "image_processor": image_processor,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))
        inputs = {
            "prompt": "horse",
            "image": init_image,
            "strength": 0.5,
            "generator": generator,
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_kandinsky_prior_emb2emb(self):

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        pipe.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())
        image = output.image_embeds
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        image_slice = image[(0), -10:]
        image_from_tuple_slice = image_from_tuple[(0), -10:]
        assert image.shape == (1, 32)
        expected_slice = np.array(
            [
                -0.83128,
                1.9208118,
                -1.0885652,
                -1.2565681,
                1.3792522,
                0.24001193,
                -2.486598,
                0.52196866,
                -1.1046519,
                0.8460418,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
