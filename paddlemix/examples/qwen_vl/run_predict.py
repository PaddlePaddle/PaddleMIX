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
import requests
from paddlenlp.trainer import PdArgumentParser
from PIL import Image, ImageDraw, ImageFont

from paddlemix import QWenLMHeadModel, QwenVLProcessor, QWenVLTokenizer
from paddlemix.utils.log import logger


def plot_boxes_to_image(image_pil, tgt):
    height, width = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        # from xywh to xyxy

        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw

        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = (
            int(x0 / 1000 * width),
            int(y0 / 1000 * height),
            int(x1 / 1000 * width),
            int(y1 / 1000 * height),
        )

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)

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

    input_image: str = field(default=None, metadata={"help": "The name of input image."})
    prompt: str = field(default=None, metadata={"help": "The prompt of the image to be generated."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="qwen-vl/qwen-vl-7b",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    seed: int = field(
        default=1234,
        metadata={"help": "random seed"},
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "output directory."},
    )
    visual: bool = field(
        default=True,
        metadata={"help": "save visual image."},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "dtype,support float32/float16/bfloat16."},
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    paddle.seed(model_args.seed)

    if model_args.dtype == "bfloat16" and not paddle.amp.is_bfloat16_supported():
        logger.warning("bfloat16 is not supported on your device,change to float32")
        model_args.dtype = "float32"

    # build tokenizer
    tokenizer = QWenVLTokenizer.from_pretrained(model_args.model_name_or_path, dtype=model_args.dtype)
    processor = QwenVLProcessor(tokenizer=tokenizer)
    # build model
    logger.info("model: {},dtypes: {}".format(model_args.model_name_or_path, model_args.dtype))
    model = QWenLMHeadModel.from_pretrained(model_args.model_name_or_path, dtype=model_args.dtype)
    model.eval()

    # input query
    query = []
    if data_args.prompt is None and data_args.input_image is None:
        raise ValueError("prompt or image must be input ")

    if data_args.input_image is not None:
        url = data_args.input_image

        # read image
        if os.path.isfile(url):
            image_pil = Image.open(data_args.input_image).convert("RGB")
        else:
            image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        query.append({"image": url})

    if data_args.prompt is not None:
        query.append({"text": data_args.prompt})

    inputs = processor(query=query, return_tensors="pd")

    pred, _ = model.generate(**inputs)
    response = processor.decode(pred)
    print("response:", response)

    boxes_ref = tokenizer._fetch_all_box_with_ref(response)

    if model_args.visual and 0 < len(boxes_ref):
        # make dir
        os.makedirs(model_args.output_dir, exist_ok=True)

        # build pred
        pred_phrases = []
        pred_boxes = []
        for obj in boxes_ref:
            if "ref" not in obj.keys():
                continue
            pred_boxes.append(list(obj["box"]))
            pred_phrases.append(obj["ref"])

        pred_dict = {
            "boxes": pred_boxes,
            "size": [image_pil.height, image_pil.width],  # H,W
            "labels": pred_phrases,
        }

        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(model_args.output_dir, "pred.jpg"))


if __name__ == "__main__":
    main()
