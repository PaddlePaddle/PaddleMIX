#!/usr/bin/env python3
from ppdet.modeling.architectures.meta_arch import BaseArch
import paddle.nn as nn
from paddle import Tensor
from ppdet.core.workspace import register


@register
class YOLOWorldDetector(BaseArch):
    """Implementation of YOLOW Series"""
    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck']
    def __init__(self,
                 backbone,
                 neck,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80) -> None:
        super().__init__()
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.backbone = backbone
        self.neck = neck
        # self.neck = YOLOWorldPAFPN()
        # self.bbox_head = YOLOWorldHead()

    def _forward(self):
        from reprod_log import ReprodLogger
        reprod_logger = ReprodLogger()
        reprod_logger.add("logits", self.inputs["image"].cpu().detach().numpy())
        reprod_logger.save("/home/onion/workspace/code/pp/Alignment/pretransform/paddle_pretransform.npy")

        img_feats, txt_feats = self.backbone(self.inputs)
        out = self.neck(img_feats, txt_feats)

        return out

    def get_pred(self):
        return self._forward()
    # def predict(self,
    #             batch_inputs,
    #             batch_data_samples,
    #             rescale: bool = True):
    #     """Predict results from a batch of inputs and data samples with post-
    #     processing.
    #     """

    #     img_feats, txt_feats = self.extract_feat(batch_inputs,
    #                                              batch_data_samples)

    #     # self.bbox_head.num_classes = self.num_test_classes
    #     self.bbox_head.num_classes = txt_feats[0].shape[0]
    #     results_list = self.bbox_head.predict(img_feats,
    #                                           txt_feats,
    #                                           batch_data_samples,
    #                                           rescale=rescale)

    #     # batch_data_samples = self.add_pred_to_datasample(
    #     #     batch_data_samples, results_list)
    #     return batch_data_samples, results_list

    # def reparameterize(self, texts) -> None:
    #     # encode text embeddings into the detector
    #     self.texts = texts
    #     self.text_feats = self.backbone.forward_text(texts)

    # def _forward(
    #         self,
    #         batch_inputs: Tensor,
    #         batch_data_samples = None):
    #     """Network forward process. Usually includes backbone, neck and head
    #     forward without any post-processing.
    #     """
    #     img_feats, txt_feats = self.extract_feat(batch_inputs,
    #                                              batch_data_samples)
    #     results = self.bbox_head.forward(img_feats, txt_feats)
    #     return results

    # def extract_feat(
    #         self, batch_inputs: Tensor,
    #         batch_data_samples):
    #     """Extract features."""
    #     txt_feats = None
    #     if batch_data_samples is None:
    #         texts = self.texts
    #         txt_feats = self.text_feats
    #     elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples:
    #         texts = batch_data_samples['texts']
    #     elif isinstance(batch_data_samples, list) and hasattr(batch_data_samples[0], 'texts'):
    #         texts = [data_sample.texts for data_sample in batch_data_samples]
    #     elif hasattr(self, 'text_feats'):
    #         texts = self.texts
    #         txt_feats = self.text_feats
    #     else:
    #         raise TypeError('batch_data_samples should be dict or list.')
    #     if txt_feats is not None:
    #         # forward image only
    #         img_feats = self.backbone.forward_image(batch_inputs)
    #     else:
    #         img_feats, txt_feats = self.backbone(batch_inputs, texts)
    #     if self.with_neck:
    #         if self.mm_neck:
    #             img_feats = self.neck(img_feats, txt_feats)
    #         else:
    #             img_feats = self.neck(img_feats)
    #     return img_feats, txt_feats
    def forword(self,
             batch_inputs,
             batch_data_samples):
        texts = batch_data_samples['texts']
        x = self.backbone(batch_inputs, texts)
        return x
