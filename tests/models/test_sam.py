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

import inspect
import unittest

import numpy as np
import paddle

from paddlemix.models.sam.configuration import SamConfig
from paddlemix.models.sam.modeling import SamModel
from paddlemix.processors.sam_processing import SamProcessor
from ppdiffusers.utils import load_image, load_numpy
from tests.models.test_modeling_common import ModelTesterMixin, floats_tensor
from tests.testing_utils import slow


class SamModelTester:
    def __init__(self, parent):
        self.parent = parent

    def get_config(self):
        # todo: more input type
        return SamConfig(input_type="boxs")

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([1, 3, 1024, 1024])
        # box prompt
        prompt = paddle.to_tensor([[[0, 0, 1024, 1024]]])
        config = self.get_config()

        return config, pixel_values, prompt

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, prompt = self.prepare_config_and_inputs()
        inputs_dict = {"img": pixel_values, "prompt": prompt}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values, prompt):
        model = SamModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values, prompt)

        self.parent.assertIsNotNone(result)


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
        self.model_tester = SamModelTester(self)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_determinism(self):
        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
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
        config = self.model_tester.get_config()
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

    @slow
    def test_model_from_pretrained(self):
        pretrained_model = "Sam/SamVitH-1024"
        model = SamModel.from_pretrained("Sam/SamVitH-1024", input_type="boxs")
        self.assertIsNotNone(model)

        # todo: check the res
        paddle.seed(1024)
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        expected_image = load_numpy(
            "https://bj.bcebos.com/v1/paddlenlp/models/community/Sam/SamVitH-1024/overture-creations-mask.npy"
        )

        image_pil = load_image(img_url)

        # build processor
        processor = SamProcessor.from_pretrained(pretrained_model)
        # build model
        input_type = "boxs"
        sam_model = SamModel.from_pretrained(pretrained_model, input_type=input_type)
        box_prompt = np.array([174, 115, 311, 465])

        image_seg, prompt = processor(
            image_pil,
            input_type=input_type,
            box=box_prompt,
            point_coords=None,
        )
        seg_masks = sam_model(img=image_seg, prompt=prompt)
        seg_masks = processor.postprocess_masks(seg_masks)

        avg_diff = np.abs(seg_masks.cpu().numpy().astype(int) - expected_image.astype(int)).mean()
        assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"

    def test_save_load(self):
        pass


if __name__ == "__main__":
    unittest.main()
