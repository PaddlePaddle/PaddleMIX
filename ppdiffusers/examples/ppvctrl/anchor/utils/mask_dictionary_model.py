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

import json
from dataclasses import dataclass, field

import paddle


@dataclass
class MaskDictionaryModel:
    mask_name: str = ""
    mask_height: int = 1080
    mask_width: int = 1920
    promote_type: str = "mask"
    labels: dict = field(default_factory=dict)

    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        updated_masks = {}
        for seg_obj_id, seg_mask in self.labels.items():
            flag = 0
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue
            for object_id, object_info in tracking_annotation_dict.labels.items():
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break
            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit

    @staticmethod
    def calculate_iou(mask1, mask2):
        mask1 = mask1.to("float32")
        mask2 = mask2.to("float32")
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        iou = intersection / union
        return iou

    def to_dict(self):
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()},
        }

    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self


@dataclass
class ObjectInfo:
    instance_id: int = 0
    mask: any = None
    class_name: str = ""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    logit: float = 0.0

    def get_mask(self):
        return self.mask

    def get_id(self):
        return self.instance_id

    def update_box(self):
        nonzero_indices = paddle.nonzero(x=self.mask)
        if nonzero_indices.shape[0] == 0:
            return []
        y_min, x_min = (paddle.min(x=nonzero_indices, axis=0), paddle.argmin(x=nonzero_indices, axis=0))[0]
        y_max, x_max = (paddle.max(x=nonzero_indices, axis=0), paddle.argmax(x=nonzero_indices, axis=0))[0]
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]

    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit,
        }
