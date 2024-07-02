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

import argparse  # noqa: F401
import os
import sys  # noqa: F401
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np  # noqa: F401
import paddle
import requests  # noqa: F401
from paddlenlp.trainer import PdArgumentParser
from PIL import Image

import paddlemix.models.imagebind as ib  # noqa: F401
from paddlemix import ImageBindModel, ImageBindProcessor
from paddlemix.datasets import *  # noqa: F401,F403
from paddlemix.models import *  # noqa: F401,F403
from paddlemix.models.imagebind.modeling import ImageBindModel  # noqa: F811
from paddlemix.models.imagebind.utils import *  # noqa: F401, F403
from paddlemix.utils.log import logger
from ppdiffusers import StableUnCLIPImg2ImgPipeline
from ppdiffusers.utils import load_image

# from paddlemix.models.imagebind.utils.resample import *
# from paddlemix.models.imagebind.utils.paddle_aux import *


ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class Predictor:
    def __init__(self, model_args):
        self.processor = ImageBindProcessor.from_pretrained(model_args.model_name_or_path)
        self.predictor = ImageBindModel.from_pretrained(model_args.model_name_or_path)
        self.predictor.eval()

    def run(self, inputs):
        with paddle.no_grad():
            embeddings = self.predictor(inputs)

        return embeddings


def main(model_args, data_args):

    # build model
    logger.info("imagebind_model: {}".format(model_args.model_name_or_path))
    url = data_args.input_image
    if os.path.isfile(url):
        # read image
        image_pil = Image.open(data_args.input_image).convert("RGB")
    elif url:
        image_pil = load_image(url)
    else:
        image_pil = None

    url = data_args.input_audio
    if os.path.isfile(url):
        # read image
        input_audio = data_args.input_audio
    elif url:
        os.system("wget {}".format(url))
        input_audio = os.path.basename(data_args.input_audio)
    else:
        input_audio = None

    predictor = Predictor(model_args)

    encoding = predictor.processor(images=image_pil, text="", audios=input_audio, return_tensors="pd")
    inputs = {}

    if image_pil:
        image_processor = encoding["pixel_values"]
        inputs.update({ModalityType.VISION: image_processor})
    if data_args.input_audio:
        audio_processor = encoding["audio_values"]
        inputs.update({ModalityType.AUDIO: audio_processor})

    embeddings = predictor.run(inputs)
    image_proj_embeds = embeddings[ModalityType.AUDIO]

    if image_pil:
        logger.info("Generate vision embedding: {}".format(embeddings[ModalityType.VISION]))
        image_proj_embeds += embeddings[ModalityType.VISION]

    if data_args.input_audio:
        logger.info("Generate audio embedding: {}".format(embeddings[ModalityType.AUDIO]))

    prompt = data_args.input_text

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_args.stable_unclip_model_name_or_path)
    pipe.set_progress_bar_config(disable=None)

    output = pipe(image_embeds=image_proj_embeds, prompt=prompt)
    os.makedirs(model_args.output_dir, exist_ok=True)

    save_path = os.path.join(model_args.output_dir, "audio2img_imagebind_output.jpg")
    logger.info("Generate image to: {}".format(save_path))
    output.images[0].save(save_path)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_text: str = field(default="", metadata={"help": "The name of prompt input."})
    input_image: str = field(
        default="",
        # wget https://github.com/facebookresearch/ImageBind/blob/main/.assets/bird_image.jpg
        metadata={"help": "The name of image input."},
    )
    input_audio: str = field(
        default="",
        # wget https://github.com/facebookresearch/ImageBind/blob/main/.assets/bird_audio.wav
        metadata={"help": "The name of audio input."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="imagebind-1.2b/",
        metadata={"help": "Path to pretrained model or model identifier"},
    )

    stable_unclip_model_name_or_path: str = field(
        default="stabilityai/stable-diffusion-2-1-unclip",
        metadata={"help": "Path to pretrained model or model identifier in stable_unclip_model_name_or_path"},
    )

    output_dir: str = field(default="vis_audio2img", metadata={"help": "The name of imagebind audio input."})

    device: str = field(
        default="GPU",
        metadata={"help": "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."},
    )


if __name__ == "__main__":

    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model_args.device = model_args.device.upper()
    assert model_args.device in ["CPU", "GPU", "XPU", "NPU"], "device should be CPU, GPU, XPU or NPU"

    main(model_args, data_args)
