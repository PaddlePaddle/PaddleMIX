# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from lcm import (
    LCMDataArguments,
    LCMModel,
    LCMModelArguments,
    LCMTrainer,
    LCMTrainingArguments,
    TextImagePair,
)
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((LCMModelArguments, LCMDataArguments, LCMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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

    model = LCMModel(model_args, training_args)
    model.set_recompute(training_args.recompute)
    model.set_xformers(training_args.enable_xformers_memory_efficient_attention)

    # only for sdxl
    ext_data_kwargs = {}
    if model_args.is_sdxl:
        ext_data_kwargs = dict(
            tokenizer_2=model.tokenizer_2,
            use_fix_crop_and_size=model_args.use_fix_crop_and_size,
            center_crop=model_args.center_crop,
            random_flip=model_args.random_flip,
        )
    train_dataset = TextImagePair(
        file_list=data_args.file_list,
        size=training_args.resolution,
        num_records=data_args.num_records,
        buffer_size=data_args.buffer_size,
        shuffle_every_n_samples=data_args.shuffle_every_n_samples,
        interpolation=data_args.interpolation,
        tokenizer=model.tokenizer,
        proportion_empty_prompts=data_args.proportion_empty_prompts,
        **ext_data_kwargs,
    )

    trainer = LCMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=model.tokenizer,
    )

    params_to_train = [p for p in model.unet.parameters() if not p.stop_gradient]
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
    # for higher ips
    main()
