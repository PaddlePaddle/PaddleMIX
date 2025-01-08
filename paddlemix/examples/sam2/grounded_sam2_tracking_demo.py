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

import argparse
import os
import sys

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
import supervision as sv
from PIL import Image

from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.processors.groundingdino_processing import GroundingDinoProcessor

sys.path.append(os.path.join(os.getcwd(), "paddlemix/models"))
from paddlemix.examples.sam2.utils.video_utils import create_video
from paddlemix.models.sam2.build_sam import build_sam2, build_sam2_video_predictor
from paddlemix.models.sam2.sam2_image_predictor import SAM2ImagePredictor

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--dino_model_name_or_path",
        type=str,
        default="GroundingDino/groundingdino-swint-ogc",
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument("--sam2_config", type=str, required=True, help="path to config file")

    parser.add_argument("--sam2_checkpoint", type=str, required=False, help="path to sam checkpoint file")

    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt of the vidoe to be segmented.")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    args = parser.parse_args()

    # build processor
    processor = GroundingDinoProcessor.from_pretrained(args.dino_model_name_or_path)
    # build model
    dino_model = GroundingDinoModel.from_pretrained(args.dino_model_name_or_path)
    dino_model.eval()

    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.sam2_config
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    inference_state = video_predictor.init_state(video_path=args.input_path)
    org_frames = inference_state["org_frames"]
    fps = inference_state["fps"]
    ann_frame_idx = 0

    image_pil = Image.fromarray(org_frames[0]).convert("RGB")
    W, H = image_pil.size
    # preprocess image text_prompt
    image_tensor, mask, tokenized_out = processor(images=image_pil, text=args.prompt)

    with paddle.no_grad():
        outputs = dino_model(
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
    filt_mask = logits_filt.max(axis=1) > args.box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # build pred
    pred_phrases = []
    pred_boxes = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = processor.decode(logit > args.text_threshold)
        pred_phrases.append(pred_phrase)

        box = box * paddle.to_tensor([W, H, W, H]).astype(paddle.float32)
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        pred_boxes.append(box)

    input_boxes = np.array(pred_boxes)
    OBJECTS = pred_phrases

    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    annotated_frames = []
    for frame_idx, segments in video_segments.items():
        img = cv2.cvtColor(org_frames[frame_idx], cv2.COLOR_BGR2RGB)
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks), mask=masks, class_id=np.array(object_ids, dtype=np.int32)
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frames.append(annotated_frame)

    create_video(annotated_frames, H, W, args.output_path, frame_rate=fps)
