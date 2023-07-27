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
from dataclasses import dataclass, field
import numpy as np
import random

import paddle
import paddle.nn as nn
from paddle.io import BatchSampler, DataLoader

from paddlenlp.ops import transfer_param
from paddlenlp.trainer import (PdArgumentParser, TrainingArguments,
                               get_last_checkpoint)
from paddlenlp.transformers import LlamaForCausalLM
from paddlevlp.datasets import load_dataset
from paddlevlp import MiniGPT4Processor, MiniGPT4ForConditionalGeneration, MiniGPT4VisionConfig, MiniGPT4QFormerConfig, MiniGPT4Config, MiniGPT4VisionModel, MiniGPT4QFormerModel
from paddlevlp.utils import paddlevlp_load
from paddlevlp.utils.log import logger
from paddlevlp.utils.parameters import freeze_parameters
from paddlevlp.optimization import CosineDecayWithWarmup


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
        default="",
        metadata={"help": "The directory path to save pretrained model or model identifier"},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained_model_path: str = field(
        default="",
        metadata={
            "help": "The path to pre-trained model that we will use for pretraining."
        },
    )
    batch_size: int = field(
        default=12,
        metadata={"help": "Number of samples in one batch."}
    )
    weight_decay: float = field(
        default=0.05, metadata={"help": "Weight decay if we apply some."}
    )
    learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate."}
    )
    num_train_epochs: float = field(
        default=200, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_start_lr: float = field(
        default=1e-6, metadata={"help": "Initial learning rate of warm up."}
    )
    eta_min: float = field(
        default=1e-5, metadata={"help": "The minimum value of learning rate."}
    )
    warmup_steps: int = field(
        default=200, metadata={"help": "Number of warmup steps."}
    )
    lr_scheduler_name: str = field(
        default="CosineDecayWithWarmup", metadata={"help": "The scheduler name to use."}
    )
    output_dir: str = field(
        default="./checkpoints", metadata={"help": "The directory name for saving checkpoint"}
    )
    use_amp: str = field(
        default=True, metadata={"help": "Whether to use amp for training."}
    )
    warmup_proportion: float = field(
        default=0.1, metadata={"help": "The warmup rate."}
    )
    freeze_vit: float = field(
        default=True, metadata={"help": "Whether to freeze vit."}
    )
    freeze_qformer: float = field(
        default=True, metadata={"help": "Whether to freeze Qformer."}
    )
    freeze_llama: float = field(
        default=True, metadata={"help": "Whether to freeze Llama."}
    )
    seed: int = field(
        default=42, metadata={"help": "The random seed."}
    )
    log_freq: int = field(
        default=1, metadata={"help":"The log frequency."}
    )
    num_workers: int = field(
        default=0, metadata={"help": "The random seed."}
    )


class MiniGPT4Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlevlp.processors.ProcessorMixin`):
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
        ValueError(
            "Cannot find pretrained model path: {}".format(pretrained_model_path)
    )

    state_dict = paddlevlp_load(pretrained_model_path, map_location="cpu")

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


def get_grouped_parameters(model, training_args):
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if p.stop_gradient:
            continue
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += paddle.numel(p)
    logger.info("number of trainable parameters: {}".format(num_parameters))
    
    grouped_params = [
        {
            "params": p_wd,
            "weight_decay": training_args.weight_decay,
        },
        {"params": p_non_wd, "weight_decay": 0.0},
    ]
    return grouped_params


def main():
    # load data, model and training parameters
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    set_seed(training_args.seed)
    paddle.set_device(training_args.device)

    # load and convert dataset
    processor = MiniGPT4Processor.from_pretrained(model_args.pretrained_model_name_or_path)
    processor.read_texts(data_args.text_path)
    minigpt4_collator = MiniGPT4Collator(processor)
    dataset = load_dataset("cc_sbu_dataset", SPLITS=["train"])
    batch_sampler = BatchSampler(dataset, batch_size=training_args.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=minigpt4_collator, num_workers=training_args.num_workers)
    
    # load MiniGPT4 model for training
    model = create_model(model_args)
    # if you wanna train from scratch, you can set del_keys = ["language_projection.weight", "language_projection.bias"]
    del_keys = []
    logger.info("Try to load the specified model.")
    load_pretrained_model(model, training_args.pretrained_model_path, del_keys=del_keys)
    logger.info("Try to convert the model dtype to the specified dtype.")
    convert_weights_to_dtype(model.vision_model, dtype="float16")
    convert_weights_to_dtype(model.qformer, dtype="float32")
    convert_weights_to_dtype(model.language_model, dtype="float16")
    logger.info("Try to freeze model parameters.")
    if training_args.freeze_vit:
        freeze_parameters(model.vision_model, enable_eval=True)
    if training_args.freeze_qformer:
        freeze_parameters(model.query_tokens)
        freeze_parameters(model.qformer, enable_eval=True)
    if training_args.freeze_llama:
        freeze_parameters(model.language_model, enable_eval=False)
    logger.info("Initializing the model done!")

    # training setting
    num_training_steps = training_args.num_train_epochs * len(train_loader)
    lr_scheduler = CosineDecayWithWarmup(
        learning_rate=training_args.learning_rate, 
        total_steps=num_training_steps,
        eta_min=training_args.eta_min, 
        warmup=training_args.warmup_steps,
        warmup_start_lr=training_args.warmup_start_lr,
        last_step=-1,
    )

    grouped_params = get_grouped_parameters(model, training_args)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=grouped_params,
        weight_decay=training_args.weight_decay,
    )
    if training_args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=65536.0, incr_every_n_steps=2000, decr_every_n_nan_or_inf=1)

    # start to train MiniGPT4
    for epoch in range(training_args.num_train_epochs):
        for step, batch_data in enumerate(train_loader):
            with paddle.amp.auto_cast(enable=training_args.use_amp, custom_white_list={}, level="O1"):
                outputs = model(**batch_data, return_dict=True)
                loss = outputs.loss
            if step % training_args.log_freq == 0:
                print("epoch: {}, step: {}, lr: {}, loss: {}".format(epoch, step, lr_scheduler.get_lr(), loss.item()))
            
            if training_args.use_amp:
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            lr_scheduler.step()
            optimizer.clear_grad()
    
    # save model
    model.save_pretrained(training_args.output_dir)
    processor.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()