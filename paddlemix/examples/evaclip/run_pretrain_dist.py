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

import numpy as np
import psutil

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 4)))
sys.path.insert(0, parent_path)
import pprint
import socket
from dataclasses import dataclass, field

import paddle
from paddlenlp.trainer import PdArgumentParser, TrainingArguments

from paddlemix.datasets import load_dataset
from paddlemix.datasets.dataset import ImageFolder
from paddlemix.metrics.clip_zero_shot import ClipZeroShot
from paddlemix.models.clip.eva_clip_model import EVACLIP, EVACLIPConfig
from paddlemix.optimization import create_optimizer
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

    task_name: str = field(
        default="coco_clip",
        metadata={
            "help": "The name of the task to use (via the datasets library), coco or laion-aes"
            " is support, if set to laion-aes, this should be the path to filelist file. "
            "option: [coco_clip/[path to laion-aes.filelist]], default: coco_clip"
        },
    )

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
        metadata={"help": "model name to create, for example [EVA02-CLIP-B-16/coca_EVA02-B-16]"},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained: bool = field(
        default=False,
        metadata={"help": "Whether to use pretrained model."},
    )
    text_wd: float = field(default=0.05, metadata={"help": "Weight decay for text tower"})
    visual_wd: float = field(default=0.05, metadata={"help": "Weight decay for visual tower"})
    text_lr: float = field(default=2e-5, metadata={"help": "The initial learning rate of text tower."})
    visual_lr: float = field(default=2e-4, metadata={"help": "The initial learning rate of visual tower."})
    layer_decay: float = field(default=1.0, metadata={"help": "The basic layer decay."})
    text_ld: float = field(default=0.75, metadata={"help": "The layer decay of text tower."})
    visual_ld: float = field(default=0.75, metadata={"help": "The layer decay of visual tower."})
    start_epoch: int = field(
        default=0,
        metadata={"help": " manual epoch number (useful on restarts)"},
    )
    context_length: int = field(
        default=77,
        metadata={"help": " context length for text."},
    )
    optimizer: str = field(default="lamb", metadata={"help": "optimizer setting, [lamb/adamw]"})
    last_epoch: int = field(default=-1, metadata={"help": "the last epoch to resume"})
    gather_with_grad: bool = field(
        default=False,
        metadata={"help": "Whether to use gather_with_grad in loss."},
    )
    local_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use local loss in loss."},
    )
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."},
    )
    tensor_fusion: bool = field(
        default=False,
        metadata={"help": "Whether to use tensor fusion."},
    )
    cpu_core_bind: bool = field(
        default=False,
        metadata={"help": "Whether to use cpu core bind."},
    )
    fuse_ln: bool = field(
        default=True,
        metadata={"help": "Whether to use fused layer norm."},
    )
    pretrained_text_model: str = field(default="openclip", metadata={"help": "the model to pre-extract text feats"})


class SelfTrainer(CLIPTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            1.0,
            num_training_steps - self.args.warmup_steps,
            last_epoch=self.args.last_epoch,
        )
        if self.args.warmup_steps > 0:
            self.lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                self.lr_scheduler,
                self.args.warmup_steps,
                0,
                1.0,
                last_epoch=self.args.last_epoch,
            )
        self.optimizer = create_optimizer(self.args, self.model, self.lr_scheduler)


class Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data_list):
        if isinstance(data_list[0], dict):
            images = [sample["image"] for sample in data_list]
            text = [sample["text"] for sample in data_list]
            batch = self.processor(
                images=images,
                text=text,
                max_length=77,
                return_tensors="pd",
                return_attention_mask=False,
                mode="train",
            )
            return batch
        else:
            images = [sample[0] for sample in data_list]
            labels = [sample[1] for sample in data_list]
            batch = self.processor(
                images=images,
                text=None,
                max_length=77,
                return_tensors="pd",
                return_attention_mask=False,
                mode="eval",
                do_resize=True,
                do_crop=True,
            )
            batch["labels"] = paddle.to_tensor(np.array(labels))
            return batch


def main_worker(training_args, model_args, data_args):
    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("bfloat16")

    config = EVACLIPConfig.from_pretrained(model_args.model)
    config["text_config"]["fusedLN"] = training_args.fuse_ln
    config["vision_config"]["fusedLN"] = training_args.fuse_ln
    model = EVACLIP(
        config,
        disable_text=False,
        local_loss=training_args.local_loss,
        gather_with_grad=training_args.gather_with_grad,
        data_world_rank=training_args.data_world_rank,
        data_world_size=training_args.data_world_size,
    )
    if training_args.pretrained:
        model.load_pretrained(model_args.model)

    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float32")

    if "laion" in data_args.task_name:
        from paddlemix.datasets.laiondata import LaionDataset

        train_dataset = LaionDataset(data_args.task_name)
    else:
        train_dataset = load_dataset(data_args.task_name, splits="train")

    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(model_args.model, "processor", "train"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(model_args.model, "processor", "train"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)
    collator = Collator(processor)

    eval_dataset = ImageFolder(f"{data_args.classification_eval}/images")
    zeroshot = ClipZeroShot(model, training_args)

    trainer = SelfTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=zeroshot.zero_shot_eval,
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    if training_args.cpu_core_bind:
        p = psutil.Process()
        start_rank = paddle.distributed.get_rank() * 3
        p.cpu_affinity([start_rank, start_rank + 1, start_rank + 2])

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)

    setdistenv(training_args)
    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    training_args.classification_eval = data_args.classification_eval
    main_worker(training_args, model_args, data_args)
