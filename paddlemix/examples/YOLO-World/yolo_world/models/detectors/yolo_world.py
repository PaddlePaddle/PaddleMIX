# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ppdet.core.workspace import create, register
from ppdet.modeling.architectures.meta_arch import BaseArch


@register
class YOLOWorldDetector(BaseArch):
    """Implementation of YOLOW Series"""

    __category__ = "architecture"

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        mm_neck=False,
        num_train_classes=80,
        num_test_classes=80,
    ):
        super().__init__()
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg["backbone"])

        # fpn
        kwargs = {"input_shape": backbone.out_shape}
        neck = create(cfg["neck"], **kwargs)

        # head
        kwargs = {"input_shape": neck.out_shape}
        bbox_head = create(cfg["bbox_head"], **kwargs)

        return {
            "backbone": backbone,
            "neck": neck,
            "bbox_head": bbox_head,
        }

    def forward(self):
        img_feats, txt_feats = self.backbone(self.inputs, self.inputs["texts"])
        if self.mm_neck:
            img_feats = self.neck(img_feats, txt_feats)
        else:
            img_feats = self.neck(img_feats)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def predict(self):

        txt_feats = None

        if self.inputs["texts"] is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(self.inputs, dict) and "texts" in self.inputs:
            texts = self.inputs["texts"]
        elif isinstance(self.inputs, list) and hasattr(self.inputs[0], "texts"):
            texts = [data_sample.texts for data_sample in self.inputs]
        elif hasattr(self, "text_feats"):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError("batch_data_samples should be dict or list.")

        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(self.inputs)
        else:
            img_feats, txt_feats = self.backbone(self.inputs, texts)

        if self.mm_neck:
            img_feats = self.neck(img_feats, txt_feats)
        else:
            img_feats = self.neck(img_feats)

        self.bbox_head.num_classes = txt_feats[0].shape[0]

        results_list = self.bbox_head.predict(
            img_feats, txt_feats, self.inputs, rescale=True
        )

        return results_list

    def get_pred(self):
        return self.predict()

    def reparameterize(self, texts):
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)
