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
import itertools
import math
import os
import pprint
import socket

import numpy as np
import paddle
from diffusion import (
    DataArguments,
    DiTDiffusionModel,
    LatentDiffusionTrainer,
    ModelArguments,
    TrainerArguments,
)
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.utils.log import logger
from transport import SiTDiffusionModel


class FeatureDataset(paddle.io.Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(
            self.labels_files
        ), "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return {"latents": features.squeeze(0), "label_id": labels.squeeze(0)}


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainerArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)
    model_args.data_world_rank = training_args.dataset_rank
    model_args.data_world_size = training_args.dataset_world_size

    training_args.report_to = ["visualdl"]
    training_args.resolution = data_args.resolution
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

    model_config_name = model_args.config_file.split("/")[-1].replace(".json", "")
    model_name = model_config_name.split("_")[0]
    assert model_name in ["DiT", "SiT", "LargeDiT"], f"Model {model_name} not supported."
    if model_name in ["DiT", "LargeDiT"]:
        model = DiTDiffusionModel(model_args, training_args)
    else:
        model = SiTDiffusionModel(model_args, training_args)
    assert model.transformer.sample_size == data_args.resolution // 8
    model.set_recompute(training_args.recompute)
    model.set_xformers(model_args.enable_xformers_memory_efficient_attention)
    model.set_ema(model_args.use_ema)

    # Setup data:
    feature_path = data_args.feature_path
    features_dir = f"{feature_path}/imagenet{data_args.resolution}_features"
    labels_dir = f"{feature_path}/imagenet{data_args.resolution}_labels"
    train_dataset = FeatureDataset(features_dir, labels_dir)

    trainer = LatentDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    # must set recompute after trainer init
    trainer.model.set_recompute(training_args.recompute)
    params_to_train = itertools.chain(trainer.model.transformer.parameters())
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
