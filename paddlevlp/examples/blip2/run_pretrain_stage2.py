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

import paddle
from paddlenlp.trainer import (PdArgumentParser, TrainingArguments,
                               get_last_checkpoint)
from paddlenlp.transformers import AutoConfig, OPTConfig, T5Config

import paddlevlp
from paddlevlp.datasets import load_dataset
from paddlevlp.models.blip2.configuration import (Blip2Config,
                                                  Blip2QFormerConfig,
                                                  Blip2VisionConfig)
from paddlevlp.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlevlp.optimization import FilterParamsName
from paddlevlp.processors.blip_processing import Blip2Processor
from paddlevlp.trainer import Trainer
from paddlevlp.utils.log import logger


class BlipCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlevlp.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        text = [sample["text_input"] for sample in data_list]
        batch = self.processor(
            images=images,
            text=text,
            max_length=32,
            return_tensors="pd",
            return_attention_mask=True,
            mode="train",
        )
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
        default="coco_caption",
        metadata={"help": "The name of the task to use (via the datasets library)."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="Salesforce/blip2-opt-2.7b",
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

    pretrained_model_path: str = field(
        default=None,
        metadata={
            "help": "The path to pre-trained model that we will use for pretraining."
        },
    )
    weight_decay: float = field(
        default=0.5, metadata={"help": "Weight decay if we apply some."}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "The initial learning rate."}
    )
    num_train_epochs: float = field(
        default=10.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_start_lr: float = field(
        default=1e-6, metadata={"help": "Initial learning rate of warm up."}
    )
    eta_min: float = field(
        default=1e-5, metadata={"help": "The minimum value of learning rate."}
    )
    warmup_steps: int = field(
        default=2000, metadata={"help": "Number of warmup steps."}
    )
    lr_scheduler_name: str = field(
        default="CosineDecayWithWarmup", metadata={"help": "The scheduler name to use."}
    )


def create_scheduler(dataset_len, config):
    lr_sched_func = getattr(paddlevlp.optimization, config.lr_scheduler_name)
    lr_sched = lr_sched_func(
        learning_rate=config.learning_rate,
        epochs=config.num_train_epochs,
        eta_min=config.eta_min,
        warmup_steps=config.warmup_steps,
        warmup_start_lr=config.warmup_start_lr,
        step_each_epoch=dataset_len,
    )
    return lr_sched


def create_optimizer_and_scheduler(model, dataset_len, config):
    lr_sched = create_scheduler(dataset_len, config)
    param_filter = FilterParamsName()
    p_wd, p_non_wd = param_filter(model)
    optimizer = paddle.optimizer.AdamW(
        parameters=p_wd + p_non_wd,
        learning_rate=lr_sched,
        weight_decay=float(config.weight_decay),
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        apply_decay_param_fun=param_filter.apply_decay_param_fun,
    )
    return optimizer, lr_sched


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
    qformer_config = Blip2QFormerConfig.from_pretrained(config.model_name_or_path)
    text_config = get_text_config(config.text_model_name_or_path)
    blip2_config = Blip2Config.from_vision_qformer_text_configs(
        vision_config, qformer_config, text_config
    )

    model = Blip2ForConditionalGeneration(blip2_config)
    return model


def load_pretrained_model(model, pretrained_model_path):
    if pretrained_model_path is None:
        return

    if not os.path.exists(pretrained_model_path):
        ValueError(
            "Cannot find pretrained model path: {}".format(pretrained_model_path)
        )

    state_dict = paddle.load(pretrained_model_path)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    model.set_state_dict(state_dict)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # if last_checkpoint is None and len(
        #         os.listdir(training_args.output_dir)) > 1:
        #     raise ValueError(
        #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #         "Use --overwrite_output_dir to overcome.")
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # create dataset
    processor = Blip2Processor.from_pretrained(model_args.model_name_or_path)
    blip_collator = BlipCollator(processor)
    train_dataset = load_dataset(data_args.task_name, splits="train")
    dataset_len = len(train_dataset)

    # create model
    model = create_model(model_args)
    load_pretrained_model(model, training_args.pretrained_model_path)

    # load model for debug
    # weight = paddle.load('/paddle/wangguanzhong/blip-jinman/PaddleNLP/blip2/blip2_checkout_4_output.pdparams')
    # model.set_state_dict(weight)

    # create optimizer
    optimizer, lr_sched = create_optimizer_and_scheduler(
        model, dataset_len, training_args
    )

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=blip_collator,
        optimizers=(optimizer, lr_sched),
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


if __name__ == "__main__":
    main()
