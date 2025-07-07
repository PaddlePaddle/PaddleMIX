# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import itertools
import os
import os.path as osp

import cv2
import paddle.nn as nn
import supervision as sv
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.cli import ArgsParser, merge_args
from tqdm import tqdm

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="image path, include image file or dir.",
    )
    parser.add_argument("--text", type=str, default=None, help="text or .txt file")
    parser.add_argument(
        "--annotation",
        action="store_true",
        help="save the annotated detection results as yolo text format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.",
    )
    parser.add_argument("--topk", default=100, type=int, help="keep topk predictions.")
    parser.add_argument(
        "--threshold",
        default=0.0,
        type=float,
        help="confidence score threshold for predictions.",
    )
    parser.add_argument("--show", action="store_true", help="show the detection results.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="first use text model to get offline text feats, then the text model will not be load when inference",
    )
    args = parser.parse_args()
    return args


def run_inference(runner, data, images, texts, max_dets, score_thr, output_dir, show, annotation):
    pred_instances = runner.model(data)[0]
    score_thr_mask = pred_instances["scores"] > score_thr
    pred_instances["scores"] = pred_instances["scores"][score_thr_mask.squeeze(-1), :].squeeze(-1)
    pred_instances["bboxes"] = pred_instances["bboxes"][score_thr_mask.squeeze(-1), :]
    pred_instances["labels"] = pred_instances["labels"][score_thr_mask.squeeze(-1), :].squeeze(-1).astype(int)

    if pred_instances["scores"].shape[0] > max_dets:
        indices = pred_instances["scores"].topk(max_dets)[1]
        pred_instances["scores"] = pred_instances["scores"][indices]
        pred_instances["bboxes"] = pred_instances["bboxes"][indices, :]
        pred_instances["labels"] = pred_instances["labels"][indices]

    for item in pred_instances.keys():
        pred_instances[item] = pred_instances[item].cpu().numpy()

    if "masks" in pred_instances:
        masks = pred_instances["masks"]
    else:
        masks = None

    image_path = images[runner.status["step_id"]]
    detections = sv.Detections(
        xyxy=pred_instances["bboxes"],
        class_id=pred_instances["labels"],
        confidence=pred_instances["scores"],
        mask=masks,
    )

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = cv2.imread(image_path)
    anno_image = image.copy()
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image)

    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(image_path)] = anno_image
        annotations_dict[osp.basename(image_path)] = detections

        ANNOTATIONS_DIRECTORY = os.makedirs(r"./annotations", exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(classes=texts, images=images_dict, annotations=annotations_dict).as_yolo(
            annotations_directory_path=ANNOTATIONS_DIRECTORY,
            min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=APPROXIMATION_PERCENTAGE,
        )

    if show:
        cv2.imshow("Image", image)  # Provide window name
        k = cv2.waitKey(0)
        if k == 27:
            # wait for ESC key to exit
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # paddle.seed(3407)
    # np.random.seed(3407)
    # random.seed(3407)
    # custom import to registry the model
    __import__("yolo_world")

    # load config
    args = parse_args()
    cfg = load_config(args.config)
    merge_args(cfg, args)
    merge_config(args.opt)

    # load text
    if cfg.text.endswith(".txt"):
        with open(cfg.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip("\r\n")] for t in lines] + [[" "]]
    else:
        texts = [[t.strip()] for t in cfg.text.split(",")] + [[" "]]

    trainer = Trainer(cfg, mode="test")

    print(trainer.model.state_dict().keys())
    trainer.load_weights(cfg.weights)

    for k, m in trainer.model.named_sublayers():
        if isinstance(m, nn.BatchNorm2D):
            m._epsilon = 1e-3  # for amp(fp16)
            m._momentum = 0.97  # 0.03 in pytorch

    trainer.model.eval()
    if cfg.offline:
        trainer.model.reparameterize([list(itertools.chain.from_iterable(texts))])

    if not osp.isfile(cfg.image):
        images = [
            osp.join(cfg.image, img) for img in os.listdir(cfg.image) if img.endswith(".png") or img.endswith(".jpg")
        ]
    else:
        images = [cfg.image]

    trainer.dataset.set_images(images)

    loader = create("TestReader")(trainer.dataset, 0)

    for step_id, data in enumerate(tqdm(loader)):
        if not cfg.offline:
            data["texts"] = [list(itertools.chain.from_iterable(texts))]
        else:
            data["texts"] = None

        trainer.status["step_id"] = step_id

        run_inference(
            trainer,
            data,
            images,
            texts,
            cfg.topk,
            cfg.threshold,
            cfg.output_dir,
            cfg.show,
            cfg.annotation,
        )
