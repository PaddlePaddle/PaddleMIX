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
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from dataclasses import dataclass, field
import yaml

import paddle
from paddlenlp.trainer import PdArgumentParser
from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlemix.utils.log import logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(
        default="http://images.cocodataset.org/val2017/000000039769.jpg", metadata={"help": "The name of input image."}
    )  # "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt: str = field(
        default=None, metadata={"help": "The prompt of the image to be generated."}
    )  # "Question: how many cats are there? Answer:"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="paddlemix/blip2-caption-opt2.7b",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    pretrained_model_path: str = field(
        default=None,
        metadata={"help": "The path to pre-trained model that we will use for inference."},
    )
    fp16: str = field(
        default=True,
        metadata={"help": "Export with mixed precision."},
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    # url = data_args.input_image  # "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # prompt = "a photo of "
    # processor = Blip2Processor.from_pretrained(model_args.model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    model.eval()
    dtype = "float32"
    if model_args.fp16:
        decorated = paddle.amp.decorate(
            models=[model.visual_encoder, model.language_model], optimizers=None, level="O2"
        )
        model.visual_encoder, model.language_model = decorated
        dtype = "float16"

    shape1 = [None, 3, None, None]
    input_spec = [
        paddle.static.InputSpec(shape=shape1, dtype="float32"),
    ]
    image_encoder = paddle.jit.to_static(model.encode_image, input_spec=input_spec)
    save_path = "blip2_export"
    paddle.jit.save(image_encoder, os.path.join(save_path, "image_encoder"))

    # TODO add test config
    deploy_info = {
        "Deploy": {
            "model": "image_encoder.pdmodel",
            "params": "image_encoder.pdiparams",
            "input_img_shape": shape1,
            "output_dtype": dtype,
        }
    }
    msg = "\n---------------Deploy Information---------------\n"
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    yml_file = os.path.join(save_path, "deploy.yaml")
    with open(yml_file, "w") as file:
        yaml.dump(deploy_info, file)

    logger.info(f"The inference model is saved in {save_path}")


if __name__ == "__main__":
    main()
