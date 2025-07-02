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

import paddle
from paddlenlp.trainer import PdArgumentParser, TrainingArguments

from paddlemix.datasets.dataset import ImageFolder
from paddlemix.examples.clip.run_pretrain_dist import Collator
from paddlemix.metrics.clip_zero_shot import ClipZeroShot
from paddlemix.models.clip.clip_model import CLIP
from paddlemix.processors.clip_processing import (
    CLIPImageProcessor,
    CLIPProcessor,
    CLIPTextProcessor,
)
from paddlemix.processors.tokenizer import SimpleTokenizer
from paddlemix.trainer import CLIPTrainer
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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="paddlemix/EVA/EVA02-CLIP-L-14",
        metadata={"help": "model name to create, for example paddlemix/EVA/EVA02-CLIP-L-14"},
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
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."},
    )


class SelfTrainer(CLIPTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.lr_scheduler = None
        self.optimizer = None


def main_worker(training_args, model_args, data_args):
    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("bfloat16")
    model = CLIP.from_pretrained(model_args.model, ignore_mismatched_sizes=False)
    model.eval()

    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float32")

    eval_dataset = ImageFolder(f"{data_args.classification_eval}/images")
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(model_args.model, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(model_args.model, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)
    collator = Collator(processor)

    zeroshot = ClipZeroShot(model, training_args)

    trainer = SelfTrainer(
        model=model, args=training_args, data_collator=collator, compute_metrics=zeroshot.zero_shot_eval
    )
    trainer.evaluate(eval_dataset=eval_dataset)


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)
    data_args.per_device_eval_batch_size = training_args.per_device_eval_batch_size
    data_args.dataloader_num_workers = training_args.dataloader_num_workers
    training_args.classification_eval = data_args.classification_eval

    setdistenv(training_args)
    main_worker(training_args, model_args, data_args)
