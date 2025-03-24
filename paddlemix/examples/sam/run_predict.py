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
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
from paddlenlp.trainer import PdArgumentParser
from PIL import Image

from paddlemix.models.sam.modeling import SamModel
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


# def show_box(box, ax, label):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
#     ax.text(x0, y0, label)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(metadata={"help": "The name of input image."})
    box_prompt: List[int] = field(default=None, metadata={"help": "box prompt format as xyxyxyxy...]."})
    points_prompt: List[int] = field(default=None, metadata={"help": "point prompt format as [[xy],[xy]...]."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="Sam/SamVitH-1024",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    input_type: str = field(
        default="boxs",
        metadata={"help": "The model prompt type, choices ['boxs', 'points', 'points_grid']."},
    )
    output_dir: str = field(
        default="seg_output",
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
    if os.path.isfile(url):
        # read image
        image_pil = Image.open(data_args.input_image).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # build processor
    processor = SamProcessor.from_pretrained(model_args.model_name_or_path)
    # build model
    logger.info("SamModel: {}".format(model_args.model_name_or_path))
    sam_model = SamModel.from_pretrained(model_args.model_name_or_path, input_type=model_args.input_type)

    if data_args.box_prompt is not None:
        data_args.box_prompt = np.array(data_args.box_prompt)
    if data_args.points_prompt is not None:
        data_args.points_prompt = np.array([data_args.points_prompt])

    image_seg, prompt = processor(
        image_pil,
        input_type=model_args.input_type,
        box=data_args.box_prompt,
        point_coords=data_args.points_prompt,
    )
    seg_masks = sam_model(img=image_seg, prompt=prompt)
    seg_masks = processor.postprocess_masks(seg_masks)

    if model_args.visual:
        # make dir
        os.makedirs(model_args.output_dir, exist_ok=True)
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pil)
        for mask in seg_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        plt.axis("off")
        plt.savefig(
            os.path.join(model_args.output_dir, "mask_pred.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )


if __name__ == "__main__":
    main()
