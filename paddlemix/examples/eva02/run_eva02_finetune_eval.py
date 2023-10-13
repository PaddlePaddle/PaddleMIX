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

import paddle
from paddlenlp.trainer import PdArgumentParser

from paddlemix.checkpoint import load_model
from paddlemix.datasets.dataset import ImageFolder
from paddlemix.examples.eva02.run_eva02_finetune_dist import (
    Collator,
    DataArguments,
    FinetuneArguments,
    ModelArguments,
)
from paddlemix.metrics.imagenet_evaluator import ImageNetEvaluator
from paddlemix.models.eva02.modeling_finetune import EVA02VisionTransformer
from paddlemix.processors.eva02_processing import (
    EVA02FinetuneImageProcessor,
    EVA02Processor,
)
from paddlemix.trainer.eva02_finetune_trainer import EVA02FinetuneTrainer
from paddlemix.utils.env import setdistenv


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


def main_worker(training_args, model_args, data_args):
    if training_args.bf16 and training_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("bfloat16")
    model = EVA02VisionTransformer.from_pretrained(model_args.model, ignore_mismatched_sizes=False)
    model.eval()

    if (
        training_args.pretrained_model_path
        and training_args.pretrained_model_path != "None"
        and training_args.resume_from_checkpoint is None
    ):
        load_model(training_args, model, ckpt_dir=training_args.pretrained_model_path)

    eval_dataset = ImageFolder(root=f"{data_args.eval_data_path}")
    image_processor = EVA02FinetuneImageProcessor.from_pretrained(os.path.join(model_args.model, "processor", "eval"))
    processor = EVA02Processor(image_processor)
    collator = Collator(processor, mode="eval")
    evaluator = ImageNetEvaluator(training_args)
    trainer = SelfTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=evaluator.clas_eval,
    )
    trainer.evaluate(eval_dataset=eval_dataset)


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
