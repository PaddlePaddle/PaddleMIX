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
    AnimateAnyoneModel_stage1,
    AnimateAnyoneTrainer_stage1,
    HumanDanceDataset,
)
from src.trainer.args_stage1 import DataArguments, ModelArguments


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.report_to = ["custom_visualdl"]

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    model = AnimateAnyoneModel_stage1(model_args)

    train_dataset = HumanDanceDataset(
        img_size=(data_args.train_width, data_args.train_height),
        data_meta_paths=data_args.meta_paths,
        sample_margin=data_args.sample_margin,
    )

    trainer = AnimateAnyoneTrainer_stage1(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    params_to_train = itertools.chain(
        list(filter(lambda p: not p.stop_gradient, trainer.model.denoising_unet.parameters())),
        list(filter(lambda p: not p.stop_gradient, trainer.model.reference_unet.parameters())),
        list(filter(lambda p: not p.stop_gradient, trainer.model.pose_guider.parameters())),
    )
    trainer.set_optimizer_grouped_parameters(params_to_train)

    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
