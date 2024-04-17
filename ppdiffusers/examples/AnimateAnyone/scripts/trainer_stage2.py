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
import itertools

import paddle
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from src.trainer import (
    AnimateAnyoneModel_stage2,
    AnimateAnyoneTrainer_stage2,
    HumanDanceVideoDataset,
)
from src.trainer.args_stage2 import DataArguments, ModelArguments


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # report to custom_visualdl
    training_args.report_to = ["custom_visualdl"]

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    model = AnimateAnyoneModel_stage2(model_args)

    train_dataset = HumanDanceVideoDataset(
        width=data_args.train_width,
        height=data_args.train_height,
        n_sample_frames=data_args.n_sample_frames,
        sample_rate=data_args.sample_rate,
        data_meta_paths=data_args.meta_paths,
    )

    trainer = AnimateAnyoneTrainer_stage2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    params_to_train = itertools.chain(
        list(filter(lambda p: not p.stop_gradient, trainer.model.denoising_unet.parameters()))
    )

    trainer.set_optimizer_grouped_parameters(params_to_train)

    # Training
    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
