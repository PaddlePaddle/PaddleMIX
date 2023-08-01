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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
import paddle.distributed as dist
from paddle.distributed import fleet
from dataclasses import dataclass, field
import numpy as np
import random
import paddle
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlenlp.trainer import (PdArgumentParser, TrainingArguments,
                               get_last_checkpoint)
from paddlenlp.transformers import AutoConfig, OPTConfig, T5Config
from paddlevlp.datasets import load_dataset
from paddlevlp.models.blip2.configuration import (
    Blip2Config, Blip2QFormerConfig, Blip2VisionConfig)
from paddlevlp.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlevlp.processors.blip_processing import Blip2Processor
from paddlevlp.trainer.blip2_trainer import BLIP2Trainer as Trainer
from paddlevlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer
from paddlevlp.models.blip2.eva_vit import interpolate_pos_embed
from paddlevlp.processors.blip_processing import BlipImageProcessor,BlipTextProcessor
from paddlevlp.examples.blip2.utils import BlipCollator,LLM_LIST,load_model

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
        metadata={
            "help": "The name of the task to use (via the datasets library)."
        }, )
    prompt: str = field(
        default="a photo of ",
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

    text_model_name_or_path: str = field(
        default="facebook/opt-2.7b",
        metadata={"help": "The type of text model to use (OPT, T5)."}, )
    image_size: int = field(
        default=224,
        metadata={"help": " Image size for training. (default:224)"})
    llm_name: str = field(
        default="opt-2.7b",
        metadata={"help": "llm name which you ned to load in LLM_LIST"})


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained_model_path: str = field(
        default="https://bj.bcebos.com/v1/paddlenlp/models/community/Salesforce/blip2-opt-2.7b/blip2_pretrained.pdparams",
        metadata={
            "help":
            "The path to pre-trained model that we will use for pretraining."
        }, )
    weight_decay: float = field(
        default=0.05, metadata={"help": "Weight decay if we apply some."})
    learning_rate: float = field(
        default=0.0001, metadata={"help": "The initial learning rate."})
    num_train_epochs: float = field(
        default=10.0,
        metadata={"help": "Total number of training epochs to perform."})
    warmup_start_lr: float = field(
        default=1e-6, metadata={"help": "Initial learning rate of warm up."})
    eta_min: float = field(
        default=1e-5, metadata={"help": "The minimum value of learning rate."})
    warmup_steps: int = field(
        default=2000, metadata={"help": "Number of warmup steps."})
    lr_scheduler_name: str = field(
        default="CosineDecayWithWarmup",
        metadata={"help": "The scheduler name to use."})
    per_device_train_batch_size: int = field(
        default=128,
        metadata={
            "help": "Batch size per GPU core/CPU for training. (default: 8)"
        })
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={
            "help": " Batch size per GPU core/CPU for evaluation. (default:8)"
        })
    warmup_start_lr: float = field(
        default=1e-6,
        metadata={"help": " The initial learning rate of blip2."})
    output_dir: str = field(default=".", metadata={"help": "The output path"})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to evaluation."})
    do_train: bool = field(default=True, metadata={"help": "Whether to train."})

    logging_steps: int = field(
        default=50, metadata={"help": "Logging interval"})
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "Evaluation strategy (epoch/steps/no)"})

    fp16_opt_level: str = field(
        default="O1", metadata={"help": "Mixed Precision Type"})
    fp16: bool = field(
        default=True, metadata={"help": "Whether to use mixed Precision"})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Forward recompute for saving graphics memory"})
    tensor_parallel_degree: int = field(
        default=1,
        metadata={"help": "Set the number of tensor model parallel"})
    sharding_parallel_degree: int = field(
        default=1,
        metadata={
            "help": "Set the number of sharding, enable sharding parallel"
        })
    pipeline_parallel_degree: int = field(
        default=1, metadata={"help": "Enable pipeline parallel"})


def get_text_config(text_model_name_or_path):
    if "t5" in text_model_name_or_path:
        text_config = T5Config.from_pretrained(text_model_name_or_path)
    elif "opt" in text_model_name_or_path:
        text_config = OPTConfig.from_pretrained(text_model_name_or_path)
    else:
        text_config = AutoConfig.from_pretrained(text_model_name_or_path)
    return text_config


def create_model(config):
    # blip2_config = Blip2ForConditionalGeneration(onfig.model_name_or_path)
    vision_config = Blip2VisionConfig.from_pretrained(config.model_name_or_path)
    qformer_config = Blip2QFormerConfig.from_pretrained(
        config.model_name_or_path)
    text_config = get_text_config(config.text_model_name_or_path)
    vision_config.image_size = config.image_size
    # add tensor_parallel_degree
    vision_config.mp_degree = config.mp_degree
    qformer_config.mp_degree = config.mp_degree
    text_config.mp_degree = config.mp_degree
    vision_config.gradient_checkpointing = config.gradient_checkpointing
    qformer_config.gradient_checkpointing = config.gradient_checkpointing
    text_config.gradient_checkpointing = config.gradient_checkpointing
    blip2_config = Blip2Config.from_vision_qformer_text_configs(
        vision_config, qformer_config, text_config)

    model = Blip2ForConditionalGeneration(blip2_config)
    paddle.device.cuda.empty_cache(
    )  # post_init_func(self, init_func, *args, **kwargs)吃显存
    return model

def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PreTrainingArguments))
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
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir) and training_args.do_train and
            not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # create dataset

    tokenizer_class = AutoTokenizer.from_pretrained(model_args.text_model_name_or_path, use_fast=False)
    image_processor = BlipImageProcessor.from_pretrained("paddlevlp/models/blip2/model_cfg/BlipImageProcessor_stage2.json")
    text_processor_class = BlipTextProcessor.from_pretrained("paddlevlp/models/blip2/model_cfg/BlipTextProcessor_stage2.json")
    processor = Blip2Processor(image_processor,text_processor_class,tokenizer_class)
    image_processor_eval = BlipImageProcessor.from_pretrained("paddlevlp/models/blip2/model_cfg/BlipImageEvalProcessor_stage2.json")
    text_processor_class_eval = BlipTextProcessor.from_pretrained("paddlevlp/models/blip2/model_cfg/BlipTextEvalProcessor_stage2.json")
    eval_processor = Blip2Processor(image_processor_eval,text_processor_class_eval,tokenizer_class)

    train_dataset = load_dataset(data_args.task_name, splits="train")
    eval_dataset = {"test": load_dataset(data_args.task_name, splits="test")}
    # create model
    blip_collator = BlipCollator(processor)
    blip_eval_collator = BlipCollator(eval_processor, mode="test")
    model_args.mp_degree = training_args.tensor_parallel_degree
    model_args.gradient_checkpointing = training_args.gradient_checkpointing
    model = create_model(model_args)

    logger.info("training_args.use_hybrid_parallel:{}".format(training_args.use_hybrid_parallel))
    # create trainer
    load_model(training_args,model, ckpt_dir="blip2_pretrained.pdparams", load_language_model=False)
    load_model(training_args,model.language_model, ckpt_dir=LLM_LIST[model_args.text_model_name_or_path])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=blip_collator,
        eval_collator=blip_eval_collator,
        processor=processor,
        eval_processor=eval_processor,
        tokenizer=tokenizer_class)
    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        state_dict = paddle.load(checkpoint)
        interpolate_pos_embed(model, state_dict)
        model.set_state_dict(state_dict)
    if training_args.do_eval:
        eval_metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


def setdistenv(args):
    if args.tensor_parallel_degree * args.sharding_parallel_degree * args.pipeline_parallel_degree != 1:
        args.use_hybrid_parallel = True
    args.dp_degree = dist.get_world_size() \
                   // (args.tensor_parallel_degree \
                    * args.sharding_parallel_degree * \
                     args.pipeline_parallel_degree)
    strategy = fleet.DistributedStrategy()
    if args.tensor_parallel_degree > 1:
        strategy.tensor_parallel = True
    args.data_parallel_degree = args.dp_degree
    logger.info("args.dp_degree:{}".format(args.dp_degree))
    logger.info("args.sharding_parallel_degree):{}".format(args.sharding_parallel_degree))
    if args.sharding_parallel_degree>1:
        args.sharding="stage1"
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
        "micro_batch_size": MICRO_BATCH_SIZE
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
    args.data_world_size = dist.get_world_size() // abs(
        args.tensor_parallel_degree * args.pipeline_parallel_degree)

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, args.data_world_rank, args.mp_rank)


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):
    device_id = paddle.device.get_device()
    assert 'gpu' in device_id

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)
    #TODO add manual_seed
    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = 1024 + basic_seed + mp_rank * 100 + data_world_rank
    global_seed = 2048 + basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


if __name__ == "__main__":
    main()
