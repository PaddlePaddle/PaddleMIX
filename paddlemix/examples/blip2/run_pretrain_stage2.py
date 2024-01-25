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
import warnings

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


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        default="coco_caption",
        metadata={"help": "The name of the task to use (via the datasets library)."},
    )
    prompt: str = field(
        default="a photo of ", metadata={"help": "The prompt of the image to be generated."}
    )  # "Question: how many cats are there? Answer:"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="paddlemix/blip2-stage2",
        metadata={"help": "Path to pretrained model or model identifier"},
    )

    text_model_name_or_path: str = field(
        default="facebook/opt-2.7b",
        metadata={"help": "The type of text model to use (OPT, T5)."},
    )
    image_size: int = field(default=224, metadata={"help": " Image size for training. (default:224)"})
    llm_name: str = field(default="opt-2.7b", metadata={"help": "llm name which you ned to load in LLM_LIST"})
    qformer_tokenizer_name: str = field(default=None, metadata={"help": "qformer tokenizer name"})

    
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
        default=32, metadata={"help": "Batch size per GPU core/CPU for training. (default: 8)"}
    )
    per_device_eval_batch_size: int = field(
        default=128, metadata={"help": " Batch size per GPU core/CPU for evaluation. (default:8)"}
    )
    warmup_start_lr: float = field(default=1e-6, metadata={"help": " The initial learning rate of blip2."})
    output_dir: str = field(default=".", metadata={"help": "The output path"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to evaluation."})
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
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
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

    @property
    def dataset_world_size(self):
        if self.sharding_parallel_degree != 1 or self.data_parallel_degree != 1:
            return max(self.sharding_parallel_degree, 1) * max(self.data_parallel_degree, 1)
        else:
            return paddle.distributed.get_world_size()


def create_model(config, training_args=None):
    blip2_config = Blip2Config.from_pretrained(config.model_name_or_path)
    blip2_config.mp_degree = config.mp_degree
    blip2_config.gradient_checkpointing = config.gradient_checkpointing
    blip2_config.qformer_config.tokenizer_name = config.qformer_tokenizer_name
    
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
    training_args.prompt = data_args.prompt
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

    # load dataset
    if isinstance(data_args.task_name, list):
        from paddlemix.datasets.dataset import ConcatDataset

        train_dataset = []
        for i in range(len(data_args.task_name)):
            train_dataset.append(load_dataset(data_args.task_name[i], splits="train"))
            train_dataset = ConcatDataset(train_dataset)
        warnings.warn("Do not support multiple {} and {} datasets.".format("test", "eval"))
        eval_dataset = {"test": load_dataset(data_args.task_name[0], splits="test")}
    else:
        train_dataset = load_dataset(data_args.task_name, splits="train")
        eval_dataset = {"test": load_dataset(data_args.task_name, splits="test")}
    # train_dataset = load_dataset(data_args.task_name,data_files=[['/root/.paddlemix/datasets/coco/images/','/root/.paddlemix/datasets/coco/annotations/coco_karpathy_train.json',"train"]])

    # create model
    blip_collator = BlipCollator(processor)
    blip_eval_collator = BlipCollator(eval_processor, mode="test")
    model_args.mp_degree = training_args.tensor_parallel_degree
    model_args.gradient_checkpointing = training_args.gradient_checkpointing
    model = create_model(model_args, training_args)

    logger.info("training_args.use_hybrid_parallel:{}".format(training_args.use_hybrid_parallel))
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
    # Training
    checkpoint = None
    if training_args.load_model_path is not None:
        load_model(training_args, model, ckpt_dir=os.path.join(training_args.load_model_path, "model_state.pdparams"))
    elif training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    if training_args.do_eval:
        eval_metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


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
