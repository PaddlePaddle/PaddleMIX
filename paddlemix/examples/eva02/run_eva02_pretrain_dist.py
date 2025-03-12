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
from paddlemix.datasets.dataset import ImageFolder
from paddlemix.models.eva02.modeling_pretrain import (
    EVA02ForPretrain,
    EVA02ForPretrainConfig,
)
from paddlemix.models.eva02.optim_factory import cosine_scheduler, create_optimizer
from paddlemix.processors.eva02_processing import (
    EVA02PretrainImageProcessor,
    EVA02Processor,
)
from paddlemix.trainer.eva02_pretrain_trainer import EVA02PretrainTrainer
from paddlemix.utils.env import setdistenv


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(
        default="",
        metadata={"help": "The dataset path."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="paddlemix/EVA/EVA02/eva02_Ti_for_pretrain",
        metadata={"help": "model name to create"},
    )
    teacher: str = field(
        default="paddlemix/EVA/EVA01-CLIP-g-14",
        metadata={"help": "teacher evaclip model name to create"},
    )
    student: str = field(
        default="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14",
        metadata={"help": "student eva02-vit model name to create"},
    )

    input_size: int = field(
        default=224,
        metadata={"help": "image size for training"},
    )
    drop: float = field(
        default=0.0,
        metadata={"help": "Dropout rate (default: 0.)"},
    )
    attn_drop_rate: float = field(
        default=0.0,
        metadata={"help": "Attention dropout rate (default: 0.)"},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate (default: 0.1)"},
    )

    # pretrain added
    layer_scale_init_value: float = field(
        default=0.0, metadata={"help": "0.1 for base, 1e-5 for large. set 0 to disable layer scale"}
    )
    rel_pos_bias: bool = field(default=False, metadata={"help": ""})
    decoupled_rel_pos_bias: bool = field(default=False, metadata={"help": ""})
    disable_decoupled_rel_pos_bias: bool = field(default=False, metadata={"help": ""})
    disable_decoupled_rel_pos_bias: bool = field(default=False, metadata={"help": ""})
    abs_pos_emb: bool = field(default=True, metadata={"help": ""})
    disable_abs_pos_emb: bool = field(default=False, metadata={"help": ""})


@dataclass
class PretrainArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    tea_pretrained_model_path: str = field(
        default=None,
        metadata={"help": "The path to pre-trained model that we will use for pretraining."},
    )
    stu_pretrained_model_path: str = field(
        default=None,
        metadata={"help": "The path to pre-trained model that we will use for pretraining."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )  # only for student
    context_length: int = field(
        default=77,
        metadata={"help": "context_length"},
    )
    only_save_updated_model: bool = field(
        default=True, metadata={"help": "Whether or not save only_save_updated_model"}
    )  # only save student eva02-vit

    optim: str = field(default="adamw", metadata={"help": "optimizer setting, [lamb/adamw]"})
    learning_rate: float = field(default=1.5e-3, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay for AdamW if we apply some."})
    weight_decay_end: float = field(default=0.05, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.98, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-6, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=3.0, metadata={"help": "Max gradient norm."})  # clip_grad

    # new added
    warmup_lr: float = field(default=1e-6, metadata={"help": "The initial learning rate for AdamW."})
    min_lr: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    warmup_steps: int = field(default=-1, metadata={"help": "Linear warmup over warmup_steps."})
    warmup_epochs: int = field(default=1, metadata={"help": "Linear warmup over warmup_epochs."})

    output_dir: str = field(
        default="output_dir",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default="output_dir/tb_pt_log",
        metadata={"help": "The output directory where logs saved."},
    )
    logging_steps: int = field(default=10, metadata={"help": "logging_steps print frequency (default: 10)"})

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(default=False, metadata={"help": "Whether to export inference model."})
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
        metadata={"help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    num_cycles: float = field(default=0.5, metadata={"help": "The number of waves in the cosine scheduler."})

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
        metadata={"help": "Whether to Use fp16 (mixed) precision instead of 32-bit"},
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

    dp_degree: int = field(
        default=2,
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

    last_epoch: int = field(default=-1, metadata={"help": "the last epoch to resume"})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=10,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )

    disable_tqdm: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."},
    )


class SelfTrainer(EVA02PretrainTrainer):
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
            warmup_steps=self.args.warmup_steps,
        )
        total_steps = int(num_training_steps_per_epoch * self.args.num_train_epochs)
        boundary = [int(x) for x in range(total_steps - 1)]
        self.lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(boundary, self.lr_schedule_values)

        self.wd_schedule_values = cosine_scheduler(
            self.args.weight_decay,
            self.args.weight_decay_end,
            self.args.num_train_epochs,
            num_training_steps_per_epoch,
        )
        print("Max WD = %.7f, Min WD = %.7f" % (max(self.wd_schedule_values), min(self.wd_schedule_values)))

        self.optimizer = create_optimizer(self.args, self.model)

        self.args.save_steps = num_training_steps_per_epoch * self.args.save_epochs


class Collator:
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
        images = [sample[0] for sample in data_list]
        # get labels from teacher's clip_features
        batch = self.processor(
            images=images,
            return_tensors="pd",
            mode=self.mode,
        )
        return batch


def main_worker(training_args, model_args, data_args):
    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("bfloat16")

    if model_args.model and model_args.model != "None":
        model_config = EVA02ForPretrainConfig.from_pretrained(model_args.model)
        # teacher_config should be str, but student_config can be a dict. if set dict, will only load config.
        if isinstance(model_config.teacher_config, str) and model_args.teacher != "None":
            model_config.teacher_config = model_args.teacher
        if isinstance(model_config.student_config, str) and model_args.student != "None":
            model_config.student_config = model_args.student
        print("model_config of teacher and student: ", model_config)
        model = EVA02ForPretrain(model_config)
    else:
        # from_pretrained, should have config and pdparams
        print("from_pretrained of teacher and student: ", model_args.teacher, model_args.student)
        model = EVA02ForPretrain.from_pretrained(
            pretrained_model_name_or_path=None,
            pretrained_teacher_name_or_path=model_args.teacher,
            pretrained_student_name_or_path=model_args.student,
        )

    if training_args.tea_pretrained_model_path and training_args.tea_pretrained_model_path != "None":
        load_model(training_args, model.teacher, ckpt_dir=training_args.tea_pretrained_model_path)

    # real pretrain from scratch student eva02-vit will not use load_model
    if training_args.stu_pretrained_model_path and training_args.stu_pretrained_model_path != "None":
        load_model(training_args, model.student, ckpt_dir=training_args.stu_pretrained_model_path)

    patch_size = model.student.get_final_patch_size()
    print("Patch size = %s" % str(patch_size))

    train_dataset = ImageFolder(root=f"{data_args.data_path}")
    image_processor = EVA02PretrainImageProcessor.from_pretrained(
        os.path.join(model_args.student, "processor", "train")
    )
    image_processor.window_size = (
        image_processor.input_size // patch_size[0],
        image_processor.input_size // patch_size[1],
    )
    processor = EVA02Processor(image_processor)
    collator = Collator(processor, mode="train")

    if paddle.distributed.get_rank() == 0:
        print("Check parameter scale !")
        for para_name, para_ver in model.student.named_parameters():
            mean = para_ver.mean().item()
            abs_mean = para_ver.abs().mean().item()
            delta = (para_ver.max() - para_ver.min()).item()
            print(
                "{}: {} {:.4f}, {:.4f}, {:.4f}, require_grad = {}".format(
                    para_name, para_ver.shape, mean, abs_mean, delta, not para_ver.stop_gradient
                )
            )

    trainer = SelfTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.save_state()


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, PretrainArguments))
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
