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
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint

from paddlemix.datasets import load_dataset
from paddlemix.models.blip2.configuration import Blip2Config
from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlemix.models.blip2.utils import BlipCollator, create_tokenizer, load_model
from paddlemix.processors.blip_processing import (
    Blip2Processor,
    BlipImageProcessor,
    BlipTextProcessor,
)
from paddlemix.trainer.blip2_trainer import BLIP2Trainer as Trainer
from paddlemix.utils.log import logger


class BlipCollator_VQA(BlipCollator):
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        if "text_input" not in data_list[0].keys():
            text = None
        else:
            text = [sample["text_input"] for sample in data_list]
        image_id = [sample["image_id"] for sample in data_list]
        question_id = [sample["question_id"] for sample in data_list]
        batch = self.processor(
            images=images,
            text=text,
            max_length=32,
            return_tensors="pd",
            return_attention_mask=True,
            mode=self.mode,
        )
        batch.update({"image_id": image_id})
        batch.update({"question_id": question_id})
        return batch


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        default="coco_vqa",
        metadata={"help": "The name of the task to use (via the datasets library)."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="paddlemix/blip2-pretrained-opt2.7b",
        metadata={"help": "Path to pretrained model or model identifier"},
    )

    text_model_name_or_path: str = field(
        default="facebook/opt-2.7b",
        metadata={"help": "The type of text model to use (OPT, T5)."},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay if we apply some."})
    learning_rate: float = field(default=0.0001, metadata={"help": "The initial learning rate."})
    num_train_epochs: float = field(default=10.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_start_lr: float = field(default=1e-6, metadata={"help": "Initial learning rate of warm up."})
    eta_min: float = field(default=1e-5, metadata={"help": "The minimum value of learning rate."})
    warmup_steps: int = field(default=2000, metadata={"help": "Number of warmup steps."})
    lr_scheduler_name: str = field(default="CosineDecayWithWarmup", metadata={"help": "The scheduler name to use."})
    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU core/CPU for training. (default: 8)"}
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": " Batch size per GPU core/CPU for evaluation. (default:8)"}
    )
    warmup_start_lr: float = field(default=1e-6, metadata={"help": " The initial learning rate of blip2."})
    output_dir: str = field(default=".", metadata={"help": "The output path"})
    do_eval: bool = field(default=True, metadata={"help": "Whether to evaluation."})
    do_train: bool = field(default=True, metadata={"help": "Whether to train."})

    logging_steps: int = field(default=50, metadata={"help": "Logging interval"})
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy (epoch/steps/no)"})

    fp16_opt_level: str = field(default="O1", metadata={"help": "Mixed Precision Type"})
    fp16: bool = field(default=True, metadata={"help": "Whether to use mixed Precision"})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Forward recompute for saving graphics memory"}
    )
    tensor_parallel_degree: int = field(default=1, metadata={"help": "Set the number of tensor model parallel"})
    sharding_parallel_degree: int = field(
        default=1, metadata={"help": "Set the number of sharding, enable sharding parallel"}
    )
    pipeline_parallel_degree: int = field(default=1, metadata={"help": "Enable pipeline parallel"})
    load_model_path: str = field(
        default=None,
        metadata={"help": "The path to model if you want to load weights from the specified path"},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark (True/False)."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options (batch_range=[10,20])."},
    )


def create_model(config, training_args=None):
    blip2_config = Blip2Config.from_pretrained(config.model_name_or_path)
    blip2_config.mp_degree = config.mp_degree
    blip2_config.gradient_checkpointing = config.gradient_checkpointing
    model = Blip2ForConditionalGeneration(blip2_config)
    model.load_pretrained(
        vision_and_bridge_name_or_path=getattr(config, "vision_and_bridge_name_or_path", None),
        vision_name_or_path=getattr(config, "vision_name_or_path", None),
        bridge_name_or_path=getattr(config, "bridge_name_or_path", None),
        training_args=training_args,
    )
    paddle.device.cuda.empty_cache()
    return model


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    setdistenv(training_args)

    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    paddle.set_device(training_args.device)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # create dataset
    tokenizer_class = create_tokenizer(model_args.text_model_name_or_path)
    if "opt" in model_args.text_model_name_or_path:
        tokenizer_class.padding_side = "left"

    image_processor = BlipImageProcessor.from_pretrained(
        os.path.join(model_args.model_name_or_path, "processor", "train")
    )
    text_processor_class = BlipTextProcessor.from_pretrained(
        os.path.join(model_args.model_name_or_path, "processor", "train")
    )
    processor = Blip2Processor(image_processor, text_processor_class, tokenizer_class)
    image_processor_eval = BlipImageProcessor.from_pretrained(
        os.path.join(model_args.model_name_or_path, "processor", "eval")
    )
    text_processor_class_eval = BlipTextProcessor.from_pretrained(
        os.path.join(model_args.model_name_or_path, "processor", "eval")
    )
    eval_processor = Blip2Processor(image_processor_eval, text_processor_class_eval, tokenizer_class)

    train_dataset = load_dataset(data_args.task_name, splits="train")
    eval_dataset = {"test": load_dataset(data_args.task_name, splits="val")}
    # create model
    blip_collator = BlipCollator(processor)
    blip_eval_collator = BlipCollator_VQA(eval_processor, mode="test")
    model_args.mp_degree = training_args.tensor_parallel_degree
    model_args.gradient_checkpointing = training_args.gradient_checkpointing
    model = create_model(model_args, training_args)
    logger.info("training_args.use_hybrid_parallel:{}".format(training_args.use_hybrid_parallel))
    # create trainer
    if training_args.load_model_path is not None:
        load_model(training_args, model, ckpt_dir=os.path.join(training_args.load_model_path, "model_state.pdparams"))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=blip_collator,
        eval_collator=blip_eval_collator,
        processor=processor,
        eval_processor=eval_processor,
        tokenizer=tokenizer_class,
    )
    eval_metrics = trainer.evaluate(eval_dataset, task_name="coco_vqa")
    trainer.log_metrics("eval", eval_metrics)


def setdistenv(args):
    if paddle.distributed.get_world_size() == 1:
        args.sharding_degree = 1
        args.tensor_parallel_degree = 1
        args.pipeline_parallel_degree = 1
        args.sharding_parallel_degree = 1

    if args.tensor_parallel_degree * args.sharding_parallel_degree * args.pipeline_parallel_degree != 1:
        args.use_hybrid_parallel = True
    args.dp_degree = dist.get_world_size() // (
        args.tensor_parallel_degree * args.sharding_parallel_degree * args.pipeline_parallel_degree
    )
    strategy = fleet.DistributedStrategy()
    if args.tensor_parallel_degree > 1:
        strategy.tensor_parallel = True
    args.data_parallel_degree = args.dp_degree
    logger.info("args.dp_degree:{}".format(args.dp_degree))
    logger.info("args.sharding_parallel_degree):{}".format(args.sharding_parallel_degree))
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.tensor_parallel_degree,
        "sharding_degree": args.sharding_parallel_degree,
        "pp_degree": args.pipeline_parallel_degree,
    }
    BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 32
    strategy.pipeline_configs = {
        "accumulate_steps": BATCH_SIZE // MICRO_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
    }
    strategy.find_unused_parameters = True

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # if paddle.distributed.get_world_size() > 1:
    #     paddle.distributed.init_parallel_env()

    args.rank = dist.get_rank()
    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    args.mp_rank = hcg.get_model_parallel_rank()
    args.dp_rank = hcg.get_data_parallel_rank()
    args.sharding_rank = hcg.get_sharding_parallel_rank()

    args.data_world_rank = args.dp_rank * args.sharding_parallel_degree + args.sharding_rank
    args.data_world_size = dist.get_world_size() // abs(args.tensor_parallel_degree * args.pipeline_parallel_degree)

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, args.data_world_rank, args.mp_rank)


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):
    device_id = paddle.device.get_device()
    assert "gpu" in device_id

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)
    # TODO add manual_seed
    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = 1024 + basic_seed + mp_rank * 100 + data_world_rank
    global_seed = 2048 + basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


if __name__ == "__main__":
    main()
