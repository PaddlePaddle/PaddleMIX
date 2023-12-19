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

from paddlemix.models.groundingdino.configuration import GroundingDinoConfig
from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.processors.groundingdino_processing import GroudingDinoProcessor

from PIL import Image
import requests

import inspect
import tempfile
import unittest

import numpy as np
import paddle
import paddle.nn as nn

from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow



class GroundingDinoModelTester:
    def __init__(self):
        pass

class GroundingDinoModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GroundingDinoModel,)
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
    
        processor = GroudingDinoProcessor.from_pretrained('GroundingDino/groundingdino-swint-ogc')

        image_tensor, mask, tokenized_out = processor(images=image_pil, text='cat')

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1) & ~np.isinf(out_1)]
            out_2 = out_2[~np.isnan(out_2) & ~np.isinf(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


        config = GroundingDinoConfig()
        
        inputs_dict = {
            'x': image_tensor,
            'm': mask,
            'input_ids': tokenized_out["input_ids"],
            'attention_mask': tokenized_out["attention_mask"],
            'text_self_attention_masks': tokenized_out["text_self_attention_masks"],
            'position_ids': tokenized_out["position_ids"],
        }
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            
            model.eval()
            with paddle.no_grad():
                input = self._prepare_for_class(inputs_dict, model_class)
                first = model(**input)
                second = model(**input)

            for k in ['pred_logits', 'pred_boxes']:
                if isinstance(first[k], tuple) and isinstance(second[k], tuple):
                    for tensor1, tensor2 in zip(first[k], second[k]):
                        check_determinism(tensor1, tensor2)
                else:
                    check_determinism(first[k], second[k])

    def test_forward_signature(self):
        config = GroundingDinoConfig()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["x", "m", "input_ids", "attention_mask", "text_self_attention_masks", 
                                  "position_ids", "targets"]
            self.assertListEqual(arg_names[:7], expected_arg_names)

    # todo: test_loadSamConfig
    def test_dinoconfig_from_pretrained(self):
        config = GroundingDinoConfig.from_pretrained("GroundingDino/groundingdino-swint-ogc")
        self.assertIsNotNone(config)


    @slow
    def test_model_from_pretrained(self):
        model = GroundingDinoModel.from_pretrained('GroundingDino/groundingdino-swint-ogc')
        self.assertIsNotNone(model)

    def test_save_load(self):
        pass

if __name__ == "__main__":
    unittest.main()
