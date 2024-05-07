# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import math
import os
import pprint
import socket

import numpy as np
import paddle
from ldm import (
    DataArguments,
    LatentDiffusionTrainer,
    ModelArguments,
    ModelC,
    MyDataset,
    TrainerArguments,
    setdistenv,
)
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.utils.log import logger

paddle.seed(1)
paddle.set_default_dtype("float32")
paddle.device.set_device("gpu")
np.random.seed(1)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainerArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)
    setdistenv(training_args)
    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size

    training_args.report_to = ["visualdl"]
    training_args.resolution = data_args.resolution
    training_args.feature_path = data_args.feature_path
    model_args.feature_path = data_args.feature_path
    training_args.benchmark = model_args.benchmark
    training_args.use_ema = model_args.use_ema
    training_args.enable_xformers_memory_efficient_attention = model_args.enable_xformers_memory_efficient_attention
    training_args.only_save_updated_model = model_args.only_save_updated_model
    training_args.profiler_options = model_args.profiler_options
    training_args.image_logging_steps = model_args.image_logging_steps = (
        (math.ceil(model_args.image_logging_steps / training_args.logging_steps) * training_args.logging_steps)
        if model_args.image_logging_steps > 0
        else -1
    )
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if training_args.seed is not None:
        set_seed(training_args.seed)

    model = ModelC(model_args, training_args)
    model.set_ema(training_args.use_ema)

    train_dataset = MyDataset(
        dataset_path=data_args.dataset_path,
        resolution=data_args.resolution,
    )

    trainer = LatentDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    params_to_train = model.unet.parameters()
    trainer.set_optimizer_grouped_parameters(params_to_train)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
