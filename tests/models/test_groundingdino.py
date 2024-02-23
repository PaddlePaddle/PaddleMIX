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
import paddle.nn.functional as F

from paddlemix.models.groundingdino.configuration import GroundingDinoConfig
from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.processors.groundingdino_processing import GroundingDinoProcessor
from ppdiffusers.utils import load_image
from tests.models.test_configuration_common import ConfigTester  # noqa: F401
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow


class GroundingDinoModelTester:
    def __init__(self, parent):
        self.parent = parent

    def get_config(self):
        return GroundingDinoConfig()

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([1, 3, 800, 800])
        mask = paddle.zeros((1, 800, 800), dtype=bool)
        tokenized_out = {
            "input_ids": ids_tensor([1, 4], 5000),
            "attention_mask": random_attention_mask([1, 4]),
            "position_ids": paddle.to_tensor([[0, 0, 1, 0]]),
            "text_self_attention_masks": random_attention_mask([1, 4, 4]),
        }

        config = self.get_config()

        return config, pixel_values, mask, tokenized_out

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, mask, tokenized_out = self.prepare_config_and_inputs()

        inputs_dict = {
            "x": pixel_values,
            "m": mask,
            "input_ids": tokenized_out["input_ids"],
            "attention_mask": tokenized_out["attention_mask"],
            "text_self_attention_masks": tokenized_out["text_self_attention_masks"],
            "position_ids": tokenized_out["position_ids"],
        }

        return config, inputs_dict

    def create_and_check_model(self, x, m, input_ids, attention_mask, text_self_attention_masks, position_ids):
        model = GroundingDinoModel(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(x, m, input_ids, attention_mask, text_self_attention_masks, position_ids)

        self.parent.assertIsNotNone(result)


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
        self.model_tester = GroundingDinoModelTester(self)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_determinism(self):
        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1) & ~np.isinf(out_1)]
            out_2 = out_2[~np.isnan(out_2) & ~np.isinf(out_2)]
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

            for k in ["pred_logits", "pred_boxes"]:
                if isinstance(first[k], tuple) and isinstance(second[k], tuple):
                    for tensor1, tensor2 in zip(first[k], second[k]):
                        check_determinism(tensor1, tensor2)
                else:
                    check_determinism(first[k], second[k])

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "x",
                "m",
                "input_ids",
                "attention_mask",
                "text_self_attention_masks",
                "position_ids",
                "targets",
            ]
            self.assertListEqual(arg_names[:7], expected_arg_names)

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(**inputs_dict)

    def test_dinoconfig_from_pretrained(self):
        config = GroundingDinoConfig.from_pretrained("GroundingDino/groundingdino-swint-ogc")
        self.assertIsNotNone(config)

    @slow
    def test_model_from_pretrained(self):
        pretrained_model_path = "GroundingDino/groundingdino-swint-ogc"
        model = GroundingDinoModel.from_pretrained(pretrained_model_path)
        self.assertIsNotNone(model)

        # test the result
        paddle.seed(1024)
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        image_pil = load_image(img_url)

        processor = GroundingDinoProcessor.from_pretrained(pretrained_model_path)
        image_tensor, mask, tokenized_out = processor(images=image_pil, text="dog")
        # expect res
        expect_boxes = np.array([0.47491184, 0.56778389, 0.26748717, 0.68284708])
        expect_label = "dog(0.78)"  # noqa: F841

        with paddle.no_grad():
            outputs = model(
                image_tensor,
                mask,
                input_ids=tokenized_out["input_ids"],
                attention_mask=tokenized_out["attention_mask"],
                text_self_attention_masks=tokenized_out["text_self_attention_masks"],
                position_ids=tokenized_out["position_ids"],
            )

        logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > 0.3
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = processor.decode(logit > 0.25)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        size = image_pil.size
        pred_dict = {  # noqa: F841
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        avg_diff = np.abs(boxes_filt[0].numpy() - expect_boxes).mean()
        self.assertLessEqual(avg_diff, 0.01)

        self.assertIn("dog", pred_phrases[0])

    def test_save_load(self):
        pass


if __name__ == "__main__":
    unittest.main()
