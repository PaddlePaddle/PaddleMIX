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

import warnings

import paddle


class SAM2Transforms(paddle.nn.Layer):
    def __init__(self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = paddle.vision.transforms.ToTensor()
        self.transforms = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.Resize((self.resolution, self.resolution)),
                paddle.vision.transforms.Normalize(self.mean, self.std),
            ]
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = paddle.stack(x=img_batch, axis=0)
        return img_batch

    def transform_coords(self, coords: paddle.Tensor, normalize=False, orig_hw=None) -> paddle.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        coords = coords * self.resolution
        return coords

    def transform_boxes(self, boxes: paddle.Tensor, normalize=False, orig_hw=None) -> paddle.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """

        boxes = self.transform_coords(boxes.reshape([-1, 2, 2]), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: paddle.Tensor, orig_hw) -> paddle.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.astype(dtype="float32")
        input_masks = masks
        mask_flat = masks.flatten(start_axis=0, stop_axis=1).unsqueeze(axis=1)
        try:
            if self.max_hole_area > 0:
                labels, areas = get_connected_components(mask_flat <= self.mask_threshold)
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape(masks.shape)
                masks = paddle.where(condition=is_hole, x=self.mask_threshold + 10.0, y=masks)
            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(mask_flat > self.mask_threshold)
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape(masks.shape)
                masks = paddle.where(condition=is_hole, x=self.mask_threshold - 10.0, y=masks)
        except Exception as e:
            warnings.warn(
                f"""{e}

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).""",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks
        masks = paddle.nn.functional.interpolate(x=masks, size=orig_hw, mode="bilinear", align_corners=False)
        return masks
