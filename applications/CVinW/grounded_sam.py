# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn.functional as F
import requests
from paddlenlp.trainer import PdArgumentParser
from PIL import Image

from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.models.sam.modeling import SamModel
from paddlemix.processors.groundingdino_processing import GroudingDinoProcessor
from paddlemix.processors.sam_processing import SamProcessor
from paddlemix.utils.log import logger


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(metadata={"help": "The name of input image."})
    prompt: str = field(default=None, metadata={"help": "The prompt of the image to be det."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dino_model_name_or_path: str = field(
        default="GroundingDino/groundingdino-swint-ogc",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    sam_model_name_or_path: str = field(
        default="Sam/SamVitH-1024",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    box_threshold: float = field(
        default=0.3,
        metadata={"help": "box threshold."},
    )
    text_threshold: float = field(
        default=0.25,
        metadata={"help": "text threshold."},
    )
    output_dir: str = field(
        default="grounded_sam_output",
        metadata={"help": "output directory."},
    )
    visual: bool = field(
        default=True,
        metadata={"help": "save visual image."},
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    url = data_args.input_image
    # bulid dino processor
    dino_processor = GroudingDinoProcessor.from_pretrained(model_args.dino_model_name_or_path)

    # bulid dino model
    logger.info("dino_model: {}".format(model_args.dino_model_name_or_path))
    dino_model = GroundingDinoModel.from_pretrained(model_args.dino_model_name_or_path)
    dino_model.eval()
    # buidl sam processor
    sam_processor = SamProcessor.from_pretrained(model_args.sam_model_name_or_path)
    # bulid model
    logger.info("SamModel: {}".format(model_args.sam_model_name_or_path))
    sam_model = SamModel.from_pretrained(model_args.sam_model_name_or_path, input_type="boxs")

    # read image
    if os.path.isfile(url):
        # read image
        image_pil = Image.open(url).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # preprocess image text_prompt
    image_tensor, mask, tokenized_out = dino_processor(images=image_pil, text=data_args.prompt)

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
    filt_mask = logits_filt.max(axis=1) > model_args.box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = dino_processor.decode(logit > model_args.text_threshold)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    logger.info("dino output{}".format(pred_dict))

    H, W = size[1], size[0]
    boxes = []
    for box in zip(boxes_filt):
        box = box[0] * paddle.to_tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box.numpy()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        boxes.append([x0, y0, x1, y1])
    boxes = np.array(boxes)
    image_seg, prompt = sam_processor(image_pil, input_type="boxs", box=boxes, point_coords=None)
    seg_masks = sam_model(img=image_seg, prompt=prompt)
    seg_masks = sam_processor.postprocess_masks(seg_masks)

    logger.info("Sam finish!")

    if model_args.visual:
        # make dir
        os.makedirs(model_args.output_dir, exist_ok=True)
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pil)
        for mask in seg_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes, pred_phrases):
            show_box(box, plt.gca(), label)

        plt.axis("off")
        plt.savefig(
            os.path.join(model_args.output_dir, "mask_pred.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

    logger.info("finish!")


if __name__ == "__main__":
    main()
