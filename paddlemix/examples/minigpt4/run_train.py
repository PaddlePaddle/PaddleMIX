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

os.environ["FLAGS_use_cuda_managed_memory"] = "true"


import random
from dataclasses import dataclass, field

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlenlp.ops import transfer_param
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import LlamaForCausalLM

from paddlemix import (
    MiniGPT4Config,
    MiniGPT4ForConditionalGeneration,
    MiniGPT4Processor,
    MiniGPT4QFormerModel,
    MiniGPT4VisionModel,
)
from paddlemix.datasets import load_dataset
from paddlemix.trainer.minigpt4_trainer import MiniGPT4Trainer as Trainer
from paddlemix.utils import paddlemix_load
from paddlemix.utils.log import logger
from paddlemix.utils.parameters import freeze_parameters


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        default="cc_sbu_dataset",
        metadata={"help": "The name of the task to use (via the datasets library)."},
    )
    text_path: str = field(
        default="data/texts.txt",
        metadata={"help": "The text file recording text used as prompt."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={"help": "The directory path to save pretrained model or model identifier"},
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
    # batch_size: int = field(
    #     default=12,
    #     metadata={"help": "Number of samples in one batch."}
    # )
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay if we apply some."})
    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate."})
    num_train_epochs: float = field(default=200, metadata={"help": "Total number of training epochs to perform."})
    warmup_start_lr: float = field(default=1e-6, metadata={"help": "Initial learning rate of warm up."})
    eta_min: float = field(default=1e-5, metadata={"help": "The minimum value of learning rate."})
    # warmup_steps: int = field(
    #     default=200, metadata={"help": "Number of warmup steps."}
    # )
    warmup: int = field(default=200, metadata={"help": "warmup ratio or steps."})
    lr_scheduler_name: str = field(default="CosineDecayWithWarmup", metadata={"help": "The scheduler name to use."})
    per_device_train_batch_size: int = field(
        default=6, metadata={"help": "Batch size per GPU core/CPU for training. (default: 8)"}
    )
    per_device_eval_batch_size: int = field(
        default=6, metadata={"help": " Batch size per GPU core/CPU for evaluation. (default:8)"}
    )
    output_dir: str = field(default="./checkpoints", metadata={"help": "The directory name for saving checkpoint"})
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

    use_amp: str = field(default=True, metadata={"help": "Whether to use amp for training."})
    warmup_proportion: float = field(default=0.1, metadata={"help": "The warmup rate."})
    freeze_vit: float = field(default=True, metadata={"help": "Whether to freeze vit."})
    freeze_qformer: float = field(default=True, metadata={"help": "Whether to freeze Qformer."})
    freeze_llama: float = field(default=True, metadata={"help": "Whether to freeze Llama."})
    seed: int = field(default=42, metadata={"help": "The random seed."})
    log_freq: int = field(default=1, metadata={"help": "The log frequency."})
    num_workers: int = field(default=0, metadata={"help": "The random seed."})

    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    model_path: str = field(
        default=None,
        metadata={"help": "The path to model if you want to load weights from the specified path"},
    )


class MiniGPT4Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        target_texts = [sample["text_input"] for sample in data_list]
        # random text from text_list read by processor and combine it with default prompt
        batch_data = self.processor(images=images, mode="train")
        target_outputs = self.processor.process_target_texts(target_texts)
        batch_data.update(target_outputs)
        return batch_data


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_model(model_args):
    config = MiniGPT4Config.from_pretrained(model_args.pretrained_model_name_or_path)
    model = MiniGPT4ForConditionalGeneration(config)
    return model


# TODO, better to split qformer, vit and llama for config and checkpoint
def load_pretrained_model(model, pretrained_model_path, del_keys=[]):
    if pretrained_model_path is None:
        return

    if not os.path.exists(pretrained_model_path):
        raise ValueError("Cannot find pretrained model path: {}".format(pretrained_model_path))

    state_dict = paddlemix_load(pretrained_model_path, map_location="cpu")

    for key in del_keys:
        state_dict.pop(key)

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                logger.warning("{}'s shape in model is not equal to the pretrained model checkpoint's".format(key))
                del state_dict[key]

    model.set_state_dict(state_dict)


def convert_weights_to_dtype(model, dtype: str):
    # trying to convert model dtype if necessary
    if dtype not in ["float16", "float32", "float64"]:
        raise ValueError("Not supported dtype: {}., only [float16, float32, float64] supported.".format(dtype))
    dtype_mapping = {
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
    }

    def convert_for_vit(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1D, nn.Conv2D)):
            if layer.weight.dtype != dtype_mapping[dtype]:
                layer.weight = transfer_param(layer.weight, restore_data=True, dtype=dtype)
            if layer.bias is not None and layer.bias.dtype != dtype_mapping[dtype]:
                layer.bias = transfer_param(layer.bias, restore_data=True, dtype=dtype)

    if isinstance(model, MiniGPT4VisionModel):
        model.apply(convert_for_vit)
    elif isinstance(model, (MiniGPT4QFormerModel, LlamaForCausalLM)):
        model.to(dtype=dtype)
    else:
        raise TypeError("Not support model type: {}.".format(type(model)))


def setdistenv(args):
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
    set_hybrid_parallel_seed(args.seed, args.data_world_rank, args.mp_rank)


def set_hybrid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):
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


def main():
    # load data, model and training parameters
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    setdistenv(training_args)

    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    model_args.mp_degree = training_args.tensor_parallel_degree
    model_args.gradient_checkpointing = training_args.gradient_checkpointing
    paddle.set_device(training_args.device)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # load and convert dataset
    processor = MiniGPT4Processor.from_pretrained(model_args.pretrained_model_name_or_path)
    processor.read_texts(data_args.text_path)
    minigpt4_collator = MiniGPT4Collator(processor)
    dataset = load_dataset("cc_sbu_dataset", SPLITS=["train"])
    # batch_sampler = BatchSampler(dataset, batch_size=training_args.batch_size, shuffle=True, drop_last=True)
    # train_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=minigpt4_collator, num_workers=training_args.num_workers)

    # load MiniGPT4 model for training
    model = create_model(model_args)

    # if you wanna train from scratch, you can set del_keys = ["language_projection.weight", "language_projection.bias"]
    # del_keys = []
    # logger.info("Try to load the specified model.")
    # load_pretrained_model(model, training_args.pretrained_model_path, del_keys=del_keys)
    # logger.info("Try to convert the model dtype to the specified dtype.")
    # convert_weights_to_dtype(model.vision_model, dtype="float16")
    # convert_weights_to_dtype(model.qformer, dtype="float32")
    # convert_weights_to_dtype(model.language_model, dtype="float16")
    # logger.info("Try to freeze model parameters.")
    if training_args.freeze_vit:
        freeze_parameters(model.vision_model, enable_eval=True)
    if training_args.freeze_qformer:
        freeze_parameters(model.query_tokens)
        freeze_parameters(model.qformer, enable_eval=True)
    if training_args.freeze_llama:
        freeze_parameters(model.language_model, enable_eval=False)
    logger.info("Initializing the model done!")

    logger.info("training_args.use_hybrid_parallel:{}".format(training_args.use_hybrid_parallel))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=minigpt4_collator,
        processor=processor,
        tokenizer=processor.tokenizer,
    )
    # Training
    checkpoint = None
    # if training_args.model_path is not None:
    #     checkpoint = training_args.model_path
    #     load_model(training_args, model, ckpt_dir=model_args.model_path, load_language_model=False)
    #     load_model(training_args, model.language_model, ckpt_dir=LLM_LIST[model_args.text_model_name_or_path])
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = os.path.join(training_args.resume_from_checkpoint, "model_state.pdparams")
    #     load_model(training_args, model, ckpt_dir=checkpoint, load_language_model=False)
    #     load_model(training_args, model.language_model, ckpt_dir=LLM_LIST[model_args.text_model_name_or_path])
    # if training_args.do_eval:
    #     eval_metrics = trainer.evaluate(eval_dataset)
    #     trainer.log_metrics("eval", eval_metrics)
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()

    # training setting
    # num_training_steps = training_args.num_train_epochs * len(train_loader)
    # lr_scheduler = CosineDecayWithWarmup(
    #     learning_rate=training_args.learning_rate,
    #     total_steps=num_training_steps,
    #     eta_min=training_args.eta_min,
    #     warmup=training_args.warmup_steps,
    #     warmup_start_lr=training_args.warmup_start_lr,
    #     last_step=-1,
    # )

    # grouped_params = get_grouped_parameters(model, training_args)
    # optimizer = paddle.optimizer.AdamW(
    #     learning_rate=lr_scheduler,
    #     parameters=grouped_params,
    #     weight_decay=training_args.weight_decay,
    # )
    # if training_args.use_amp:
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=65536.0, incr_every_n_steps=2000, decr_every_n_nan_or_inf=1)

    # # start to train MiniGPT4
    # for epoch in range(training_args.num_train_epochs):
    #     for step, batch_data in enumerate(train_loader):
    #         with paddle.amp.auto_cast(enable=training_args.use_amp, custom_white_list={}, level="O1"):
    #             outputs = model(**batch_data, return_dict=True)
    #             loss = outputs.loss
    #         if step % training_args.log_freq == 0:
    #             print("epoch: {}, step: {}, lr: {}, loss: {}".format(epoch, step, lr_scheduler.get_lr(), loss.item()))

    #         if training_args.use_amp:
    #             scaled = scaler.scale(loss)
    #             scaled.backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #         else:
    #             loss.backward()
    #             optimizer.step()

    #         lr_scheduler.step()
    #         optimizer.clear_grad()

    # # save model
    # model.save_pretrained(training_args.output_dir)
    # processor.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
