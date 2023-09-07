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
from typing import Optional

import paddle
from paddlenlp.trainer import PdArgumentParser, TrainingArguments

from paddlemix.checkpoint import load_model
from paddlemix.datasets.imagenet.datasets_build import build_dataset
from paddlemix.metrics.imagenet_utils import MetricLogger, accuracy
from paddlemix.models.eva02.modeling_finetune import EVA02VisionTransformer
from paddlemix.trainer.eva02_finetune_trainer import EVA02FinetuneTrainer
from paddlemix.utils.env import setdistenv


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_set: str = field(
        default="IMNET",  # "image_folder"
        metadata={"help": "ImageNet dataset path."},
    )
    data_path: str = field(
        default="/paddle/dataset/ILSVRC2012/train",
        metadata={"help": "The dataset path."},
    )
    eval_data_path: str = field(
        default="/paddle/dataset/ILSVRC2012/val",
        metadata={"help": "ImageNet dataset path."},
    )
    nb_classes: int = field(
        default=1000,
        metadata={"help": "ImageNet dataset path."},
    )
    imagenet_default_mean_and_std: bool = field(
        default=False,
        metadata={"help": "ImageNet dataset path."},
    )

    # Augmentation parameters
    color_jitter: float = field(
        default=0.4,
        metadata={"help": "Color jitter factor (default: 0.4)"},
    )
    aa: str = field(
        default="rand-m9-mstd0.5-inc1",
        metadata={"help": 'Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'},
    )
    scale_low: float = field(
        default=0.08,
        metadata={"help": "[scale_low, 1.0]"},
    )
    train_interpolation: str = field(
        default="bicubic",
        metadata={"help": 'Training interpolation (random, bilinear, bicubic default: "bicubic")'},
    )
    no_aug: bool = field(default=False, metadata={"help": "no_aug"})
    reprob: float = field(default=0.0, metadata={"help": "Random erase prob (default: 0.25)"})
    remode: str = field(default="pixel", metadata={"help": 'Random erase mode (default: "pixel")'})
    recount: int = field(default=1, metadata={"help": "Random erase count (default: 1)"})
    resplit: bool = field(default=False, metadata={"help": "Do not random erase first (clean) augmentation split"})

    # Evaluation parameters
    crop_pct: float = field(default=1.0, metadata={"help": "Evaluation crop param for data aug."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14",
        metadata={"help": "model name to create"},
    )
    input_size: int = field(
        default=336,
        metadata={"help": "image size for training"},
    )


@dataclass
class FinetuneArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained_model_path: str = field(
        default=None,
        metadata={"help": "The path to pre-trained model that we will use for pretraining."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

    output_dir: str = field(
        default="output_dir",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default="output_dir/tb_ft_log",
        metadata={"help": "The output directory where logs saved."},
    )
    logging_steps: int = field(default=10, metadata={"help": "logging_steps print frequency (default: 10)"})

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(default=False, metadata={"help": "Whether to export infernece model."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU core/CPU for training."})
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU core/CPU for evaluation."}
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=10,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )

    dp_degree: int = field(
        default=1,
        metadata={"help": " data parallel degrees."},
    )
    sharding_parallel_degree: int = field(
        default=1,
        metadata={"help": " sharding parallel degrees."},
    )
    tensor_parallel_degree: int = field(
        default=1,
        metadata={"help": " tensor parallel degrees."},
    )
    pipeline_parallel_degree: int = field(
        default=1,
        metadata={"help": " pipeline parallel degrees."},
    )
    sharding_degree: int = field(
        default=1,
        metadata={"help": ("@deprecated Please use sharding_parallel_degree. ")},
    )

    disable_tqdm: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."},
    )


class SelfTrainer(EVA02FinetuneTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.lr_scheduler = None
        self.optimizer = None


@paddle.no_grad()
def evaluate(data_loader, model, args):
    criterion = paddle.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]  # [384, 3, 336, 336]
        target = batch[-1]  # [384]
        target = target.cast("int64")

        with paddle.amp.auto_cast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main_worker(training_args, model_args, data_args):
    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("bfloat16")
    model = EVA02VisionTransformer.from_pretrained(model_args.model, ignore_mismatched_sizes=False)
    model.eval()

    training_args.model = model_args.model
    if (
        training_args.pretrained_model_path
        and training_args.pretrained_model_path != "None"
        and training_args.resume_from_checkpoint is None
    ):
        load_model(training_args, model, ckpt_dir=training_args.pretrained_model_path)
    # if training_args.bf16 and training_args.fp16_opt_level == "O2":
    #     paddle.set_default_dtype("float32")

    data_args.input_size = model_args.input_size
    dataset_val, _ = build_dataset(is_train=False, args=data_args)
    data_loader_val = paddle.io.DataLoader(
        dataset_val,
        batch_size=int(training_args.per_device_eval_batch_size),
        num_workers=training_args.dataloader_num_workers,
    )

    test_stats = evaluate(data_loader_val, model, data_args)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, FinetuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)

    setdistenv(training_args)
    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    main_worker(training_args, model_args, data_args)
