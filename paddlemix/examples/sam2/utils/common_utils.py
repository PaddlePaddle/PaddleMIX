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
import os
import random

import cv2
import numpy as np
import supervision as sv


class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)

        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            mask_npy_path = os.path.join(mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy")
            mask = np.load(mask_npy_path)
            unique_ids = np.unique(mask)
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0:
                    continue
                else:
                    object_mask = mask == uid
                    all_object_masks.append(object_mask[None])
            if len(all_object_masks) == 0:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            file_path = os.path.join(json_path, "mask_" + raw_image_name.split(".")[0] + ".json")
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            object_id_to_name = {}
            with open(file_path, "r") as file:
                json_data = json.load(file)
                for obj_id, obj_item in json_data["labels"].items():
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids:
                        continue
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    all_object_boxes.append([x1, y1, x2, y2])
                    class_name = obj_item["class_name"]
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
                    object_id_to_name[instance_id] = class_name
            paired_id_and_box = zip(all_object_ids, all_object_boxes)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=all_object_masks,
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)

        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            mask_npy_path = os.path.join(mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy")
            mask = np.load(mask_npy_path)
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = 0, 0, 0
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
            file_path = os.path.join(json_path, "mask_" + raw_image_name.split(".")[0] + ".json")
            with open(file_path, "r") as file:
                json_data = json.load(file)
                for obj_id, obj_item in json_data["labels"].items():
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)
                print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
