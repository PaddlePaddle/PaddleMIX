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

import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import paddle
import paddlenlp
from emu3.mllm import Emu3Config, Emu3ForCausalLM, Emu3Tokenizer
from emu3.train.datasets import Emu3FeatureDataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)


@dataclass
class TrainingArguments(paddlenlp.transformers.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)


def update_configs(model_config, args, fields):
    cross_update = (
        lambda a, b, field_name: setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None
        else setattr(a, field_name, getattr(b, field_name))
    )
    for f in fields:
        cross_update(model_config, args, f)


def train():
    parser = paddlenlp.transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    os.environ["WANDB_DIR"] = os.path.join(training_args.output_dir, "wandb")
    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
        torch_dtype="bfloat16" if training_args.bf16 else None,
    )
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )
    train_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer)
    trainer = paddlenlp.transformers.Trainer(model=model, args=training_args, train_dataset=train_dataset)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    paddle.device.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
