# coding:utf-8

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

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 4)))
sys.path.insert(0, parent_path)
import pprint
import socket
from dataclasses import dataclass, field

from paddlenlp.trainer import PdArgumentParser, TrainingArguments

from paddlemix.checkpoint import load_model
from paddlemix.datasets.laion_clip import get_data
from paddlemix.metrics.clip_zero_shot import zero_shot_eval
from paddlemix.models.evaclip.eva_clip_model import EVACLIP
from paddlemix.processors.clip_processing import image_transform
from paddlemix.utils.env import setdistenv


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    classification_eval: str = field(
        default="",
        metadata={"help": "Path to IN1K data."},
    )

    precomputed_text_emb: str = field(
        default="open_clip_vit_g_14",
        metadata={"help": "precomputed_text_emb name."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="paddlemix/EVA/EVA02-CLIP-L-14",
        metadata={"help": "model name to create, for example paddlemix/EVA/EVA02-CLIP-L-14"},
    )
    model_name_or_path: str = field(
        default="clip",
        metadata={"help": "Path to pretrained model or model identifier"},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained_model_path: str = field(
        default=None,
        metadata={"help": "The path to pre-trained model that we will use for pretraining."},
    )
    pretrained_text_model: str = field(default="openclip", metadata={"help": "the model to pre-extract text feats"})


def evaluate(model, dataloader_dict, args):
    model.eval()
    ret = zero_shot_eval(model, dataloader_dict, args)
    model.train()
    return ret


def main_worker(training_args, model_args, data_args):
    model = EVACLIP.from_pretrained(model_args.model, ignore_mismatched_sizes=False)

    training_args.model = model_args.model
    if (
        training_args.pretrained_model_path
        and training_args.pretrained_model_path != "None"
        and training_args.resume_from_checkpoint is None
    ):
        load_model(training_args, model, ckpt_dir=training_args.pretrained_model_path)

    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    dataloader_dict = get_data(data_args, (preprocess_train, preprocess_val))

    evaluate(model, dataloader_dict, training_args)


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)
    data_args.per_device_eval_batch_size = training_args.per_device_eval_batch_size
    data_args.dataloader_num_workers = training_args.dataloader_num_workers

    setdistenv(training_args)
    main_worker(training_args, model_args, data_args)
