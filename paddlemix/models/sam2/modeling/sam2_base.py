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

import paddle

from paddlemix.models.sam2.modeling.sam2_utils import (
    MLP,
    get_1d_sine_pe,
    select_closest_cond_frames,
)
from paddlemix.models.sam2.modeling.sam.mask_decoder import MaskDecoder
from paddlemix.models.sam2.modeling.sam.prompt_encoder import PromptEncoder
from paddlemix.models.sam2.modeling.sam.transformer import TwoWayTransformer

NO_OBJ_SCORE = -1024.0
init_trunc_normal = paddle.nn.initializer.TruncatedNormal(std=0.02)


class SAM2Base(paddle.nn.Layer):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            self.mask_downsample = paddle.nn.Conv2D(in_channels=1, out_channels=1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            self.mem_dim = tuple(self.memory_encoder.out_proj.weight.shape)[0]
        self.num_maskmem = num_maskmem
        self.maskmem_tpos_enc = paddle.create_parameter(
            shape=paddle.zeros(shape=[num_maskmem, 1, 1, self.mem_dim]).shape,
            dtype=paddle.zeros(shape=[num_maskmem, 1, 1, self.mem_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[num_maskmem, 1, 1, self.mem_dim])),
        )
        self.maskmem_tpos_enc.stop_gradient = False

        init_trunc_normal(self.maskmem_tpos_enc)

        self.no_mem_embed = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, self.hidden_dim]).shape,
            dtype=paddle.zeros(shape=[1, 1, self.hidden_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, self.hidden_dim])),
        )
        self.no_mem_embed.stop_gradient = not True

        self.no_mem_pos_enc = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, self.hidden_dim]).shape,
            dtype=paddle.zeros(shape=[1, 1, self.hidden_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, self.hidden_dim])),
        )
        self.no_mem_pos_enc.stop_gradient = not True

        init_trunc_normal(self.no_mem_embed)
        init_trunc_normal(self.no_mem_pos_enc)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = paddle.create_parameter(
                shape=paddle.zeros(shape=[1, self.hidden_dim]).shape,
                dtype=paddle.zeros(shape=[1, self.hidden_dim]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, self.hidden_dim])),
            )
            self.no_obj_ptr.stop_gradient = not True

            init_trunc_normal(self.no_obj_ptr)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = paddle.create_parameter(
                shape=paddle.zeros(shape=[1, self.mem_dim]).shape,
                dtype=paddle.zeros(shape=[1, self.mem_dim]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, self.mem_dim])),
            )
            self.no_obj_embed_spatial.stop_gradient = not True
            init_trunc_normal(self.no_obj_embed_spatial)
        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuningSee notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **self.sam_mask_decoder_extra_args or {},
        )
        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = paddle.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = paddle.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = paddle.nn.Linear(in_features=self.hidden_dim, out_features=self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = paddle.nn.Identity()

    def _forward_sam_heads(
        self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.shape[0]

        assert backbone_features.shape[1] == self.sam_prompt_embed_dim
        assert backbone_features.shape[2] == self.sam_image_embedding_size
        assert backbone_features.shape[3] == self.sam_image_embedding_size
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.shape[0] == B and sam_point_labels.shape[0] == B
        else:
            sam_point_coords = paddle.zeros(shape=[B, 1, 2])
            sam_point_labels = -paddle.ones(shape=[B, 1], dtype="int32")
        if mask_inputs is not None:
            assert len(tuple(mask_inputs.shape)) == 4 and tuple(mask_inputs.shape)[:2] == (B, 1)
            if tuple(mask_inputs.shape)[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = paddle.nn.functional.interpolate(
                    mask_inputs.astype(dtype="float32"),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
        )
        (low_res_multimasks, ious, sam_output_tokens, object_score_logits) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = paddle.where(
                condition=is_obj_appearing[:, None, None], x=low_res_multimasks, y=NO_OBJ_SCORE
            )
        low_res_multimasks = low_res_multimasks.astype(dtype="float32")
        high_res_multimasks = paddle.nn.functional.interpolate(
            x=low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = paddle.argmax(x=ious, axis=-1)
            batch_inds = paddle.arange(end=B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(axis=1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(axis=1)
            if sam_output_tokens.shape[1] > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = (low_res_multimasks, high_res_multimasks)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.astype(dtype="float32")
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.astype(dtype="float32")
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = paddle.nn.functional.interpolate(
            high_res_masks,
            size=(high_res_masks.shape[-2] // 4, high_res_masks.shape[-1] // 4),
            align_corners=False,
            mode="bilinear",
        )
        ious = paddle.ones(shape=[mask_inputs.shape[0], 1], dtype=mask_inputs.dtype).astype(dtype="float32")
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = paddle.zeros(shape=[mask_inputs.shape[0], self.hidden_dim])
        else:
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        is_obj_appearing = paddle.any(x=mask_inputs.flatten(start_axis=1).astype(dtype="float32") > 0.0, axis=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.astype(dtype="float32")
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return (low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits)

    def forward_image(self, img_batch: paddle.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(tuple(x.shape)[-2], tuple(x.shape)[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(start_axis=2).transpose(perm=[2, 0, 1]) for x in feature_maps]
        vision_pos_embeds = [x.flatten(start_axis=2).transpose(perm=[2, 0, 1]) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].shape[1]
        C = self.hidden_dim
        H, W = feat_sizes[-1]

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].transpose(perm=[1, 2, 0]).reshape([B, C, H, W])
            return pix_feat
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        if not is_init_cond_frame:

            to_cat_memory, to_cat_memory_pos_embed = [], []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                elif not track_in_reverse:
                    prev_frame_idx = (frame_idx - 2) // stride * stride
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                else:
                    prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)

                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)

                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue

                feats = prev["maskmem_features"]

                to_cat_memory.append(feats.flatten(start_axis=2).transpose(perm=[2, 0, 1]))
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.flatten(start_axis=2).transpose(perm=[2, 0, 1])
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    (
                        (frame_idx - t) * tpos_sign_mul
                        if self.use_signed_tpos_enc_to_obj_ptrs
                        else abs(frame_idx - t),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or num_frames is not None and t >= num_frames:
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = paddle.stack(x=ptrs_list, axis=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = paddle.to_tensor(data=pos_list)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(axis=1).expand(shape=[-1, B, self.mem_dim])
                    else:
                        obj_pos = paddle.zeros(shape=[len(pos_list), B, self.mem_dim], dtype=obj_ptrs.dtype)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.transpose(perm=[0, 2, 1, 3]).flatten(start_axis=0, stop_axis=1)
                        obj_pos = obj_pos.repeat_interleave(repeats=C // self.mem_dim, axis=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = tuple(obj_ptrs.shape)[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.transpose(perm=[1, 2, 0]).view(B, C, H, W)
                return pix_feat_with_mem

            to_cat_memory = [self.no_mem_embed.expand(shape=[1, B, self.mem_dim])]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(shape=[1, B, self.mem_dim])]

        memory = paddle.concat(x=to_cat_memory, axis=0)
        memory_pos_embed = paddle.concat(x=to_cat_memory_pos_embed, axis=0)
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.transpose(perm=[1, 2, 0]).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].shape[1]
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].transpose(perm=[1, 2, 0]).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).astype(dtype="float32")
        else:
            mask_for_mem = paddle.nn.functional.sigmoid(x=pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).astype(dtype="float32")
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[
                ..., None, None
            ].expand(shape=tuple(maskmem_features.shape))
        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.transpose(perm=[1, 2, 0]).view(x.shape[1], x.shape[2], *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].transpose(perm=[1, 2, 0])
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=point_inputs is not None,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        (_, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = object_score_logits
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.shape[0]
        if batch_size == 1:
            return pred_masks

        max_obj_inds = paddle.argmax(x=pred_masks, axis=0, keepdim=True)
        batch_obj_inds = paddle.arange(end=batch_size)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = paddle.where(condition=keep, x=pred_masks, y=paddle.clip(x=pred_masks, max=-10.0))
        return pred_masks
