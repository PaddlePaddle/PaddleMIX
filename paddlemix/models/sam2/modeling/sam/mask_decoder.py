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

from typing import List, Optional, Tuple, Type

import paddle

from paddlemix.models.sam2.modeling.sam2_utils import MLP, LayerNorm2d


class MaskDecoder(paddle.nn.Layer):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: paddle.nn.Layer,
        num_multimask_outputs: int = 3,
        activation: Type[paddle.nn.Layer] = paddle.nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = paddle.nn.Embedding(num_embeddings=1, embedding_dim=transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = paddle.nn.Embedding(num_embeddings=self.num_mask_tokens, embedding_dim=transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = paddle.nn.Embedding(num_embeddings=1, embedding_dim=transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.output_upscaling = paddle.nn.Sequential(
            paddle.nn.Conv2DTranspose(
                in_channels=transformer_dim, out_channels=transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            paddle.nn.Conv2DTranspose(
                in_channels=transformer_dim // 4, out_channels=transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = paddle.nn.Conv2D(
                in_channels=transformer_dim, out_channels=transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = paddle.nn.Conv2D(
                in_channels=transformer_dim, out_channels=transformer_dim // 4, kernel_size=1, stride=1
            )
        self.output_hypernetworks_mlps = paddle.nn.LayerList(
            sublayers=[
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = paddle.nn.Linear(in_features=transformer_dim, out_features=1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: paddle.Tensor,
        image_pe: paddle.Tensor,
        sparse_prompt_embeddings: paddle.Tensor,
        dense_prompt_embeddings: paddle.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: paddle.Tensor,
        image_pe: paddle.Tensor,
        sparse_prompt_embeddings: paddle.Tensor,
        dense_prompt_embeddings: paddle.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        s = 0
        if self.pred_obj_scores:
            output_tokens = paddle.concat(
                x=[self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], axis=0
            )
            s = 1
        else:
            output_tokens = paddle.concat(x=[self.iou_token.weight, self.mask_tokens.weight], axis=0)
        output_tokens = output_tokens.unsqueeze(axis=0).expand(shape=[sparse_prompt_embeddings.shape[0], -1, -1])
        tokens = paddle.concat(x=(output_tokens, sparse_prompt_embeddings), axis=1)
        if repeat_image:
            src = paddle.repeat_interleave(x=image_embeddings, repeats=tuple(tokens.shape)[0], axis=0)
        else:
            assert tuple(image_embeddings.shape)[0] == tuple(tokens.shape)[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.shape[0] == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = paddle.repeat_interleave(x=image_pe, repeats=tuple(tokens.shape)[0], axis=0)
        b, c, h, w = tuple(src.shape)
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]
        x = src
        perm_53 = list(range(x.ndim))
        perm_53[1] = 2
        perm_53[2] = 1
        src = x.transpose(perm=perm_53).reshape([b, c, h, w])
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        hyper_in_list: List[paddle.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = paddle.stack(x=hyper_in_list, axis=1)
        b, c, h, w = tuple(upscaled_embedding.shape)
        masks = (hyper_in @ upscaled_embedding.reshape([b, c, h * w])).reshape([b, -1, h, w])
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * paddle.ones(shape=[tuple(iou_pred.shape)[0], 1], dtype=iou_pred.dtype)
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(start_axis=-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = paddle.sum(x=mask_logits > stability_delta, axis=-1).astype(dtype="float32")
        area_u = paddle.sum(x=mask_logits > -stability_delta, axis=-1).astype(dtype="float32")
        stability_scores = paddle.where(condition=area_u > 0, x=area_i / area_u, y=1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = paddle.argmax(x=multimask_iou_scores, axis=-1)
        batch_inds = paddle.arange(end=multimask_iou_scores.shape[0])
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(axis=1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(axis=1)
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh
        mask_logits_out = paddle.where(
            condition=is_stable[..., None, None].expand_as(y=singlemask_logits),
            x=singlemask_logits,
            y=best_multimask_logits,
        )
        iou_scores_out = paddle.where(
            condition=is_stable.expand_as(y=singlemask_iou_scores),
            x=singlemask_iou_scores,
            y=best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
