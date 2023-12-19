# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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




sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from PIL import Image
from paddlemix.models.sam.modeling import SamModel
from paddlemix.models.sam.configuration import SamConfig
from paddlemix.processors.sam_processing import SamProcessor
import requests


import inspect
import tempfile
import unittest

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.transformers.opt.configuration import OPTConfig

from paddlemix.models.blip2 import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerConfig,
    Blip2VisionConfig,
)
from paddlemix.models.blip2.eva_vit import VisionTransformer
from paddlemix.models.blip2.modeling import BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST
from paddlemix.models.blip2.Qformer import BertLMHeadModel
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow


class SamModelTester:
    def __init__(self):
        pass

class SamModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SamModel,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        pass
        # self.model_tester = SamModelTester(self)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_determinism(self):
        # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        url = '/home/aistudio/work/PaddleMIX/paddlemix/examples/Sam/overture-creations.png'
        if os.path.isfile(url):
            # read image
            image_pil = Image.open(url).convert("RGB")
        else:
            image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            
        processor = SamProcessor.from_pretrained('Sam/SamVitH-1024')
                    
        image_seg, prompt = processor(
            image_pil,
            input_type='boxs',
            box=np.array([0, 0, 512, 512]),
            point_coords=None,
        )

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        # model = SamModel.from_pretrained('Sam/SamVitH-1024', input_type='boxs')
        # seg_masks = model(img=image_seg, prompt=prompt)

        config = SamConfig(input_type='boxs')
        inputs_dict = {
            'img': floats_tensor([1, 3, 1024, 1024]),
            'prompt': prompt
        }
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                input = self._prepare_for_class(inputs_dict, model_class)
                first = model(**input)
                second = model(**input)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_forward_signature(self):
        config = SamConfig(input_type='boxs')
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["img", "prompt"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_sam_config_from_pretrained(self):
        config = SamConfig.from_pretrained("Sam/SamVitH-1024")
        self.assertIsNotNone(config)


    #todo @slow
    def test_model_from_pretrained(self):
        model = SamModel.from_pretrained('Sam/SamVitH-1024', input_type='boxs')
        self.assertIsNotNone(model)

    def test_save_load(self):
        pass

if __name__ == "__main__":
    unittest.main()
