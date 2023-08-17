# coding:utf-8
import sys
import os
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
import pprint
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import paddle
from IPython import embed
import math
import numpy as np
from paddlemix.datasets.imagenet.datasets_build import build_dataset
from paddlemix.models.eva02.modeling_finetune import EVA02VisionTransformer
from paddlemix.models.eva02.optim_factory import create_optimizer
from paddlemix.checkpoint import save, load_model
from paddlemix.utils.env import setdistenv
from paddlemix.trainer.eva02_finetune_trainer import EVA02FinetuneTrainer
from paddlenlp.trainer import (PdArgumentParser, TrainingArguments)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    data_set: str = field(
        default="IMNET", # "image_folder"
        metadata={"help": "ImageNet dataset path."}, )
    data_path: str = field(
        default="/paddle/dataset/ILSVRC2012/train",
        metadata={"help": "The dataset path."}, )
    eval_data_path: str = field(
        default="/paddle/dataset/ILSVRC2012/val",
        metadata={"help": "ImageNet dataset path."}, )
    nb_classes: int = field(
        default=1000,
        metadata={"help": "ImageNet dataset path."}, )
    imagenet_default_mean_and_std: bool = field(
        default=False,
        metadata={"help": "ImageNet dataset path."}, )

    # Augmentation parameters
    color_jitter: float = field(
        default=0.4,
        metadata={"help": 'Color jitter factor (default: 0.4)'}, )
    aa: str = field(
        default="rand-m9-mstd0.5-inc1",
        metadata={"help": 'Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'}, )
    scale_low: float = field(
        default=0.08,
        metadata={"help": '[scale_low, 1.0]'}, )
    train_interpolation: str = field(
        default='bicubic',
        metadata={"help": 'Training interpolation (random, bilinear, bicubic default: "bicubic")'}, )
    no_aug: bool = field(default=False, metadata={"help": 'no_aug'})
    reprob: float = field(default=0.0, metadata={"help": 'Random erase prob (default: 0.25)'})
    remode: str = field(default='pixel', metadata={"help": 'Random erase mode (default: "pixel")'})
    recount: int = field(default=1, metadata={"help": 'Random erase count (default: 1)'})
    resplit: bool = field(default=False, metadata={"help": 'Do not random erase first (clean) augmentation split'})

    # Evaluation parameters
    crop_pct: str = field(default=1.0, metadata={"help": 'Evaluation crop param for data aug.'})

 
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model: str = field(
        default="eva02_config/EVA02/EVA02-B-14/",
        metadata={"help": "model name to create"}, )
    input_size: int = field(
        default=336,
        metadata={"help": "image size for training"}, )

    layer_decay: float = field(default=0.9, metadata={"help": "layer_decay."})
    drop: float = field(
        default=0.,
        metadata={"help": "Dropout rate (default: 0.)"}, )
    attn_drop_rate: float = field(
        default=0.,
        metadata={"help": "Attention dropout rate (default: 0.)"}, )
    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate (default: 0.1)"}, )

    model_ema: bool = field(default=False, metadata={"help": "enable ema training"})
    model_ema_decay: float = field(default=0.9999, metadata={"help": "ema decay"})

    smoothing: float = field(default=0.1, metadata={"help": "Label smoothing (default: 0.1)"})
    linear_probe: bool = field(default=False, metadata={"help": "linear_probe"})


@dataclass
class FinetuneArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """
    pretrained_model_path: str = field(
        default=None,
        metadata={"help": "The path to pre-trained model that we will use for pretraining."}, )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    # context_length: int = field(
    #     default=77,
    #     metadata={"help": "context_length"}, )

    optim: str = field(default="adamw", metadata={"help": "optimizer setting, [lamb/adamw]"})
    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay for AdamW if we apply some."})
    weight_decay_end: float = field(default=0.05, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=0.0, metadata={"help": "Max gradient norm."}) # clip_grad

    # new added
    warmup_lr: float = field(default=0.0, metadata={"help": "The initial learning rate for AdamW."})
    min_lr: float = field(default=0.0, metadata={"help": "The initial learning rate for AdamW."})
    warmup_steps: int = field(default=-1, metadata={"help": "Linear warmup over warmup_steps."})
    warmup_epochs: int = field(default=1, metadata={"help": "Linear warmup over warmup_epochs."})

    output_dir: str = field(
        default="output_dir", metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default="output_dir/tb_ft_log", metadata={"help": "The output directory where logs saved."},
    )
    logging_steps: int = field(
        default=10, metadata={"help": "logging_steps print frequency (default: 10)"})

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(default=False, metadata={"help": "Whether to export infernece model."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU core/CPU for training."})
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    accum_freq: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    num_train_epochs: float = field(default=100, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. suppor linear, cosine, constant, constant_with_warmup"},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    num_cycles: float = field(default=0.5, metadata={"help": "The number of waves in the cosine scheduler."})
    lr_end: float = field(default=1e-7, metadata={"help": "The end LR in the polynomial scheduler."})
    power: float = field(default=1.0, metadata={"help": "The power factor in the polynomial scheduler."})

    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_epochs: int = field(default=1, metadata={"help": "Save checkpoint every X updates epochs."})

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (no_cuda). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: AMP optimization level selected in ['O0', 'O1', and 'O2']. "
                "See details at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html"
            )
        },
    )

    dp_degree: int = field(default=2, metadata={"help": " data parallel degrees."}, )
    sharding_parallel_degree: int = field(default=1, metadata={"help": " sharding parallel degrees."}, )
    tensor_parallel_degree: int = field(default=1, metadata={"help": " tensor parallel degrees."}, )
    pipeline_parallel_degree: int = field(default=1, metadata={"help": " pipeline parallel degrees."}, )
    sharding_degree: int = field(default=1, metadata={"help": ("@deprecated Please use sharding_parallel_degree. ")},)

    last_epoch: int = field(
        default=-1, metadata={"help": "the last epoch to resume"})

    dataloader_drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=10,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."}, )


class SelfTrainer(EVA02FinetuneTrainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        total_train_batch_size = self.args.train_batch_size * self.args.accum_freq * self.args.dataset_world_size
        num_training_steps_per_epoch = len(self.train_dataset) // total_train_batch_size
        self.lr_schedule_values = cosine_scheduler(
            self.args.learning_rate,
            self.args.min_lr,
            self.args.num_train_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=self.args.warmup_epochs,
            warmup_steps=self.args.warmup_steps, )
        total_steps = int(num_training_steps_per_epoch * self.args.num_train_epochs)
        boundary = [int(x) for x in range(total_steps-1)]
        self.lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(boundary, self.lr_schedule_values)

        self.wd_schedule_values = cosine_scheduler(
            self.args.weight_decay, self.args.weight_decay_end, self.args.num_train_epochs,
            num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" %
            (max(self.wd_schedule_values), min(self.wd_schedule_values)))

        num_layers = self.model.get_num_layers()
        if self.args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(
                list(self.args.layer_decay**(num_layers + 1 - i)
                    for i in range(num_layers + 2)))
        else:
            assigner = None
        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = self.model.no_weight_decay()
        self.optimizer = create_optimizer(
            self.args,
            self.model,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)

        self.args.save_steps = num_training_steps_per_epoch * self.args.save_epochs


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1,
                     sched_type="cos"):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    if sched_type == "cos":
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) *
            (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
        ])
    elif sched_type == "linear":
        schedule = np.linspace(base_value, final_value,
                               epochs * niter_per_ep - warmup_iters)
    else:
        raise NotImplementedError()

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


def main_worker(training_args, model_args, data_args):
    if training_args.bf16 and training_args.fp16_opt_level == 'O2':
        paddle.set_default_dtype("bfloat16")
    model = EVA02VisionTransformer.from_pretrained(
        model_args.model,
        ignore_mismatched_sizes=True)

    training_args.model = model_args.model
    if training_args.pretrained_model_path and training_args.pretrained_model_path != "None" and training_args.resume_from_checkpoint is None:
        load_model(
            training_args, model, ckpt_dir=training_args.pretrained_model_path)

    data_args.input_size = model_args.input_size
    data_args.linear_probe = model_args.linear_probe
    train_dataset, data_args.nb_classes = build_dataset(is_train=True, args=data_args)

    training_args.smoothing = model_args.smoothing
    training_args.linear_probe = model_args.linear_probe
    training_args.layer_decay = model_args.layer_decay
    trainer = SelfTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,)

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


if __name__ == "__main__":
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, FinetuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)

    training_args.gradient_accumulation_steps = training_args.accum_freq

    setdistenv(training_args)

    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    main_worker(training_args, model_args, data_args)
