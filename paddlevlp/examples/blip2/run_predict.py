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
import sys
import os
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from dataclasses import dataclass, field
import paddle
import requests
from paddlenlp.trainer import PdArgumentParser
from PIL import Image

from paddlevlp.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlevlp.processors.blip_processing import Blip2Processor
from paddlevlp.utils.log import logger
from paddlevlp.examples.blip2.utils import load_pretrained_model


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(metadata={
        "help": "The name of input image."
    })  # "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt: str = field(
        default=None,
        metadata={"help": "The prompt of the image to be generated."
                  })  # "Question: how many cats are there? Answer:"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="Salesforce/blip2-opt-2.7b",
        metadata={"help": "Path to pretrained model or model identifier"}, )
    pretrained_model_path: str = field(
        default=None,
        metadata={
            "help":
            "The path to pre-trained model that we will use for inference."
        }, )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    url = (data_args.input_image
           )  # "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt = data_args.prompt
    processor = Blip2Processor.from_pretrained(
        model_args.model_name_or_path)  # "Salesforce/blip2-opt-2.7b"
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pd",
        return_attention_mask=True,
        mode="test", )
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path)

    # load checkpoint
    if model_args.pretrained_model_path:
        weight = paddle.load(model_args.pretrained_model_path)
        model.set_state_dict(weight)

    model.eval()
    model.to("gpu")  # doctest: +IGNORE_RESULT
    generated_ids, scores = model.generate(**inputs)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()
    logger.info("Generate text: {}".format(generated_text))


if __name__ == "__main__":
    main()
