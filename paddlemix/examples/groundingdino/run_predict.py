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

import numpy as np
import paddle
import paddle.nn.functional as F
import requests
from paddlenlp.trainer import PdArgumentParser
from PIL import Image, ImageDraw, ImageFont

from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.processors.groundingdino_processing import GroundingDinoProcessor
from paddlemix.utils.log import logger


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * paddle.to_tensor([W, H, W, H]).astype(paddle.float32)
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box.numpy()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(metadata={"help": "The name of input image."})
    prompt: str = field(default=None, metadata={"help": "The prompt of the image to be generated."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="GroundingDino/groundingdino-swint-ogc",
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
        default="output",
        metadata={"help": "output directory."},
    )
    visual: bool = field(
        default=True,
        metadata={"help": "save visual image."},
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # build processor
    processor = GroundingDinoProcessor.from_pretrained(model_args.model_name_or_path)
    # build model
    logger.info("dino_model: {}".format(model_args.model_name_or_path))
    dino_model = GroundingDinoModel.from_pretrained(model_args.model_name_or_path)
    dino_model.eval()
    # read image
    url = data_args.input_image
    # read image
    if os.path.isfile(url):
        # read image
        image_pil = Image.open(data_args.input_image).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # preprocess image text_prompt
    image_tensor, mask, tokenized_out = processor(images=image_pil, text=data_args.prompt)

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
        pred_phrase = processor.decode(logit > model_args.text_threshold)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    logger.info("output{}".format(pred_dict))

    if model_args.visual:
        # make dir
        os.makedirs(model_args.output_dir, exist_ok=True)
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(model_args.output_dir, "pred.jpg"))


if __name__ == "__main__":
    main()
