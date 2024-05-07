# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Optional, Tuple, Union, Sequence

# from examples.YOLO_World.yolo_world.models.utils.util import filter_scores_and_topk
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from ppdet.modeling.bbox_utils import batch_distance2bbox

from ..utils import BaseConv, multi_apply

class ContrastiveHead(nn.Layer):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """

    def __init__(self) -> None:
        super().__init__()

        self.bias = self.create_parameter(shape=[], default_initializer=nn.initializer.Constant(value=0.0), is_bias=True)
        self.logit_scale = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Assign(np.ones(1) * np.log(1 / 0.07)))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, axis=1, p=2)
        w = F.normalize(w, axis=-1, p=2)

        batch, channel, height, width = x.shape
        _, k, _ = w.shape
        x = paddle.transpose(x, perm=[0, 2, 3, 1]) # bchw->bhwc
        x = x.reshape([batch, -1, channel])  # bhwc->b(hw)c
        w = paddle.transpose(w, perm=[0, 2, 1])  # bkc->bck
        x = paddle.matmul(x, w)

        x = x.reshape([batch, height, width, k])
        x = paddle.transpose(x, perm=[0, 3, 1, 2])

        x = x * paddle.exp(self.logit_scale) + self.bias
        return x


class BNContrastiveHead(nn.Layer):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int) -> None:
        super().__init__()

        self.norm = nn.BatchNorm2D(embed_dims)

        self.bias = self.create_parameter(shape=[], default_initializer=nn.initializer.Constant(value=0.0), is_bias=True)

        # use -1.0 is more stable
        self.logit_scale = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Assign(paddle.full([1], -1.0)))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, axis=-1, p=2)

        batch, channel, height, width = x.shape
        _, k, _ = w.shape

        x = paddle.transpose(x, perm=[0, 2, 3, 1])  # bchw->bhwc
        x = x.reshape([batch, -1, channel])  # bhwc->b(hw)c
        w = paddle.transpose(w, perm=[0, 2, 1])  # bkc->bck
        x = paddle.matmul(x, w)
        x = x.reshape([batch, height, width, k])
        x = paddle.transpose(x, perm=[0, 3, 1, 2])

        x = x * paddle.exp(self.logit_scale) + self.bias
        return x


class YOLOWorldHeadModule(nn.Layer):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """

    def __init__(self,
                 in_channels: List,
                 num_classes: int,
                 reg_max: int,
                 embed_dims: int,
                 use_bn_head: bool = False) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_max = reg_max
        super().__init__()


    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.cls_contrasts = nn.LayerList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for in_c in self.in_channels:
            self.reg_preds.append(
                nn.Sequential(
                    BaseConv(in_channels=in_c,
                               out_channels=reg_out_channels,
                               ksize=3,
                               stride=1),
                    BaseConv(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               ksize=3,
                               stride=1),
                    nn.Conv2D(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    BaseConv(in_channels=in_c,
                               out_channels=cls_out_channels,
                               ksize=3,
                               stride=1),
                    BaseConv(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               ksize=3,
                               stride=1),
                    nn.Conv2D(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead())

        self.proj: Tensor = paddle.arange(self.reg_max, dtype=paddle.float32)


    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor):
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(len(self.in_channels))]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.LayerList, reg_pred: nn.LayerList,
                       cls_contrast: nn.LayerList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = paddle.transpose(paddle.reshape(
                bbox_dist_preds, [-1, 4, self.reg_max, h * w]), [0, 3, 1, 2])

            bbox_preds = paddle.squeeze(paddle.matmul(
                F.softmax(bbox_dist_preds, axis=3), paddle.unsqueeze(self.proj.reshape([-1, 1]), axis=-1)), axis=-1)
            bbox_preds = paddle.transpose(paddle.reshape(bbox_preds, (b, -1, h, w)), [0, 2, 3, 1])
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


class YOLOWorldHead(nn.Layer):
    """YOLO-World Head
    """

    def __init__(self,
                 in_channels: List,
                 num_classes: int,
                 reg_max: int,
                 embed_dims: int,
                 fpn_strides,
                 grid_cell_offset: float = 0.5,
                 use_bn_head: bool = False) -> None:
        super().__init__()

        self.fpn_strides = fpn_strides
        self.grid_cell_offset = grid_cell_offset
        self.head_module = YOLOWorldHeadModule(
            in_channels,
            num_classes,
            reg_max,
            embed_dims,
            use_bn_head
        )

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples,
                rescale: bool = False):
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions


    def _generate_anchors(self, feats_size: List[tuple], dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            h, w = feats_size[i]
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg = None,
                        rescale: bool = True,
                        with_nms: bool = True):
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        flatten_priors, flatten_stride  = self._generate_anchors(featmap_sizes)
        flatten_cls_scores = [
            paddle.transpose(cls_score, perm=[0, 2, 3, 1]).reshape([num_imgs, -1, self.num_classes])
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            paddle.transpose(bbox_pred, perm=[0, 2, 3, 1]).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = F.sigmoid(paddle.concat(flatten_cls_scores, axis=1))
        flatten_bbox_preds = paddle.concat(flatten_bbox_preds, axis=1)

        # flatten_decoded_bboxes = self.bbox_coder.decode(
        #     flatten_priors[None], flatten_bbox_preds, flatten_stride)
        flatten_decoded_bboxes = batch_distance2bbox(flatten_priors, flatten_bbox_preds)
        flatten_decoded_bboxes *= flatten_stride

        if with_objectnesses:
            flatten_objectness = [
                paddle.transpose(objectness, perm=[0, 2, 3, 1]).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = F.sigmoid(paddle.concat(flatten_objectness, axis=1))
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = dict()
                empty_results["bboxes"] = bboxes
                empty_results["scores"] = scores[:, 0]
                empty_results["labels"] = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = dict()
            results["scores"]=scores
            results["labels"]=labels
            results["bboxes"]=bboxes[keep_idxs]

            if rescale:
                if pad_param is not None:
                    results["bboxes"] -= results["bboxes"].new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results["bboxes"] /= results["bboxes"].new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            # nms
            # results = self._bbox_post_process(results=results,
            #                                   cfg=cfg,
            #                                   rescale=False,
            #                                   with_nms=with_nms,
            #                                   img_meta=img_meta)

            results["bboxes"][:, 0::2].clamp_(0, ori_shape[1])
            results["bboxes"][:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
