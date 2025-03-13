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

import copy
from typing import Tuple

import numpy as np
import paddle

from paddlemix.models.sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs), key=lambda x: abs(x - frame_idx)
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = paddle.arange(dtype="float32", end=pe_dim)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(axis=-1) / dim_t
    pos_embed = paddle.concat(x=[pos_embed.sin(), pos_embed.cos()], axis=-1)
    return pos_embed


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return paddle.nn.functional.relu
    if activation == "gelu":
        return paddle.nn.functional.gelu
    if activation == "glu":
        return paddle.nn.functional.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for i in range(N)])


class DropPath(paddle.nn.Layer):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = paddle.to_tensor(1 - self.drop_prob)
        shape = (tuple(x.shape)[0],) + (1,) * (x.ndim - 1)

        random_tensor = paddle.empty(shape=shape, dtype=x.dtype).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.divide_(y=keep_prob)
        return x * random_tensor


class MLP(paddle.nn.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: paddle.nn.Layer = paddle.nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.LayerList(
            sublayers=(
                paddle.nn.Linear(in_features=n, out_features=k) for n, k in zip([input_dim] + h, h + [output_dim])
            )
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = paddle.nn.functional.sigmoid(x=x)
        return x


class LayerNorm2d(paddle.nn.Layer):
    def __init__(self, num_channels: int, eps: float = 1e-06) -> None:
        super().__init__()
        out_9 = paddle.create_parameter(
            shape=paddle.ones(shape=num_channels).shape,
            dtype=paddle.ones(shape=num_channels).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=num_channels)),
        )
        out_9.stop_gradient = not True
        self.weight = out_9
        out_10 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_channels).shape,
            dtype=paddle.zeros(shape=num_channels).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=num_channels)),
        )
        out_10.stop_gradient = not True
        self.bias = out_10
        self.eps = eps

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(axis=1, keepdim=True)
        s = (x - u).pow(y=2).mean(axis=1, keepdim=True)
        x = (x - u) / paddle.sqrt(x=s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: paddle.Tensor,
    noise: float = 0.1,
    noise_bound: int = 20,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixels), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserved for top left and 3 for bottom right corners, dtype=torch.int32
    """

    box_coords = mask_to_box(masks)
    B, _, H, W = tuple(masks.shape)
    box_labels = paddle.to_tensor(data=[top_left_label, bottom_right_label], dtype="int32").repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, paddle.Tensor):
            noise_bound = paddle.to_tensor(data=noise_bound)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = paddle.min(bbox_w * noise, noise_bound)
        max_dy = paddle.min(bbox_h * noise, noise_bound)
        box_noise = 2 * paddle.rand(shape=[B, 1, 4]) - 1
        box_noise = box_noise * paddle.stack(x=(max_dx, max_dy, max_dx, max_dy), axis=-1)
        box_coords = box_coords + box_noise
        img_bounds = paddle.to_tensor(data=[W, H, W, H]) - 1
        box_coords.clip_(min=paddle.zeros_like(x=img_bounds), max=img_bounds)
    box_coords = box_coords.reshape(-1, 2, 2)
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:
        pred_masks = paddle.zeros_like(x=gt_masks)
    assert gt_masks.dtype == "bool" and gt_masks.shape[1] == 1
    assert pred_masks.dtype == "bool" and tuple(pred_masks.shape) == tuple(gt_masks.shape)
    assert num_pt >= 0
    B, _, H_im, W_im = tuple(gt_masks.shape)

    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    all_correct = paddle.all(x=(gt_masks == pred_masks).flatten(start_axis=2), axis=2)
    all_correct = all_correct[..., None, None]
    pts_noise = paddle.rand(shape=[B, num_pt, H_im, W_im, 2])
    pts_noise[..., 0] *= fp_masks | all_correct & ~gt_masks
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(start_axis=2).argmax(axis=2)
    labels = (pts_idx % 2).to("int32")
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = paddle.stack(x=[pts_x, pts_y], axis=2).to("float32")
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = paddle.zeros_like(x=gt_masks)
    assert gt_masks.dtype == "bool" and gt_masks.shape[1] == 1
    assert pred_masks.dtype == "bool" and tuple(pred_masks.shape) == tuple(gt_masks.shape)
    B, _, _, W_im = tuple(gt_masks.shape)

    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = paddle.zeros(shape=[B, 1, 2], dtype="float32")
    labels = paddle.ones(shape=[B, 1], dtype="int32")
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im
        points[b, 0, 1] = pt_idx // W_im
        labels[b, 0] = int(is_positive)

    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")
