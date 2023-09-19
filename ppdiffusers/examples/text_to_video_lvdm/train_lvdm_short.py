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
import itertools
import math
import os

import paddle
from lvdm import LatentVideoDiffusion, LatentVideoDiffusionTrainer, VideoFrameDataset
from lvdm.lvdm_args_short import (
    ModelArguments,
    TrainerArguments,
    VideoFrameDatasetArguments,
)
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser(
        (
            ModelArguments,
            VideoFrameDatasetArguments,
            TrainerArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        trainer_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    # report to custom_visualdl
    training_args.report_to = ["custom_visualdl"]
    training_args.resolution = data_args.resolution

    training_args.image_logging_steps = trainer_args.image_logging_steps = (
        (math.ceil(trainer_args.image_logging_steps / training_args.logging_steps) * training_args.logging_steps)
        if trainer_args.image_logging_steps > 0
        else -1
    )

    training_args.print_config(model_args, "Model")
    training_args.print_config(trainer_args, "Trainer")
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

    model = LatentVideoDiffusion(model_args)

    train_dataset = VideoFrameDataset(
        data_root=data_args.train_data_root,
        resolution=data_args.resolution,
        video_length=data_args.video_length,
        dataset_name=data_args.dataset_name,
        subset_split=data_args.train_subset_split,
        spatial_transform=data_args.spatial_transform,
        clip_step=data_args.clip_step,
        temporal_transform=data_args.temporal_transform,
    )
    eval_dataset = VideoFrameDataset(
        data_root=data_args.eval_data_root,
        resolution=data_args.resolution,
        video_length=data_args.video_length,
        dataset_name=data_args.dataset_name,
        subset_split=data_args.eval_subset_split,
        spatial_transform=data_args.spatial_transform,
        clip_step=data_args.clip_step,
        temporal_transform=data_args.temporal_transform,
    )

    trainer = LatentVideoDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # must set recompute after trainer init
    trainer.model.set_recompute(training_args.recompute)

    params_to_train = itertools.chain(trainer.model.unet.parameters())
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
