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
import paddle.distributed as dist
from diffusion import (
    AutoTrainerArguments,
    DataArguments,
    DiTDiffusionModelAuto,
    LatentDiffusionAutoTrainer,
    ModelArguments,
    shard_w,
)
from paddlenlp.data import Stack
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.utils.log import logger

# from transport import SiTDiffusionModel

MODEL_CLASSES = {
    "DiT": DiTDiffusionModelAuto,
    "LargeDiT": DiTDiffusionModelAuto,
    # to do SiT support auto parallel
    # "SiT": SiTDiffusionModelAuto,
}


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


def collate_data(data, stack_fn=Stack()):
    latents = stack_fn([x["latents"] for x in data])
    label_id = stack_fn([x["label_id"] for x in data])

    return {
        "latents": [latents, label_id],
        "label_id": label_id,  # for dynamic to static, must be 2 fields
    }


def shard_model(model):
    """
    shard the model and initialize each parameter

    Args:
        model (paddle.nn.Layer): Neural network model of PaddlePaddle

    Returns:
        None

    Raises:
        None
    """
    pp_stage = 0
    for name, layer in model.named_sublayers(include_self=False):
        if name.startswith("transformer"):
            if hasattr(layer, "pp_stage"):
                pp_stage = layer.pp_stage

            if any(
                n in name
                for n in ["attention.wq", "attention.wk", "attention.wv", "feed_forward.w1", "feed_forward.w3"]
            ):
                layer.weight = shard_w(layer.weight, pp_stage, [dist.Replicate(), dist.Shard(1)])

            if any(n in name for n in ["attention.wo", "feed_forward.w2"]):
                layer.weight = shard_w(layer.weight, pp_stage, [dist.Replicate(), dist.Shard(0)])

            # To be determined
            if any(n in name for n in ["attention.q_norm", "attention.k_norm"]):
                if hasattr(layer, "weight"):
                    layer.weight = shard_w(layer.weight, pp_stage, [dist.Replicate(), dist.Replicate()])
                    layer.bias = shard_w(layer.bias, pp_stage, [dist.Replicate(), dist.Replicate()])

            if any(n in name for n in ["attention_norm", "ffn_norm"]):
                layer.weight = shard_w(layer.weight, pp_stage, [dist.Replicate(), dist.Replicate()])

            if any(n in name for n in ["adaLN_modulation.1"]) and "final_layer" not in name:
                layer.weight = shard_w(layer.weight, pp_stage, [dist.Replicate(), dist.Shard(1)])
                layer.bias = shard_w(layer.bias, pp_stage, [dist.Replicate(), dist.Shard(0)])

            if any(n in name for n in ["x_embedder", "t_embedder.mlp.0"]):
                # first pp stage
                if hasattr(layer, "weight") and "proj" not in name:
                    layer.weight = shard_w(layer.weight, 0, [dist.Replicate(), dist.Shard(1)])
                    layer.bias = shard_w(layer.bias, 0, [dist.Replicate(), dist.Shard(0)])
                elif hasattr(layer, "proj"):
                    layer.proj.weight = shard_w(layer.proj.weight, 0, [dist.Replicate(), dist.Replicate()])
                    layer.proj.bias = shard_w(layer.proj.bias, 0, [dist.Replicate(), dist.Replicate()])

            if any(n in name for n in ["t_embedder.mlp.2"]):
                # first pp stage
                layer.weight = shard_w(layer.weight, 0, [dist.Replicate(), dist.Shard(0)])
                layer.bias = shard_w(layer.bias, 0, [dist.Replicate(), dist.Replicate()])

            if "y_embedder.embedding_table" in name:
                # first pp stage
                layer.weight = shard_w(layer.weight, 0, [dist.Replicate(), dist.Replicate()])

            if any(n in name for n in ["final_layer.linear", "final_layer.adaLN_modulation.1"]):
                # last pp stage
                layer.weight = shard_w(layer.weight, -1, [dist.Replicate(), dist.Shard(1)])
                layer.bias = shard_w(layer.bias, -1, [dist.Replicate(), dist.Shard(0)])
            # print(f"pp_stage {pp_stage} layer {name}")

        elif name.startswith("vae"):
            if any(n in name for n in ["vae.encoder", "vae.quant_conv"]):
                if hasattr(layer, "weight"):
                    # first pp stage
                    # print(f"pp_stage 0 layer {name}")
                    layer.weight = shard_w(layer.weight, 0, [dist.Replicate(), dist.Replicate()])
                    layer.bias = shard_w(layer.bias, 0, [dist.Replicate(), dist.Replicate()])

            if any(n in name for n in ["vae.decoder", "vae.post_quant_conv"]):
                if hasattr(layer, "weight"):
                    # last pp stage
                    # print(f"pp_stage -1 layer {name}")
                    layer.weight = shard_w(layer.weight, -1, [dist.Replicate(), dist.Replicate()])
                    layer.bias = shard_w(layer.bias, -1, [dist.Replicate(), dist.Replicate()])


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, AutoTrainerArguments))
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
    model_class = MODEL_CLASSES[model_name]
    model = model_class(model_args, training_args)  # and initialized model

    assert model.transformer.sample_size == data_args.resolution // 8
    model.set_recompute(training_args.recompute)
    model.set_xformers(model_args.enable_xformers_memory_efficient_attention)
    model.set_ema(model_args.use_ema)
    shard_model(model)
    # todo optimization: init_model(model) after shard_model during cold start

    # Setup data:
    feature_path = data_args.feature_path
    features_dir = f"{feature_path}/imagenet{data_args.resolution}_features"
    labels_dir = f"{feature_path}/imagenet{data_args.resolution}_labels"
    train_dataset = FeatureDataset(features_dir, labels_dir)

    trainer = LatentDiffusionAutoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_data,
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
