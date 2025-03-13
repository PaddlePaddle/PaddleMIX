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
from dataclasses import dataclass, field
import json
from typing import List, Optional,Tuple,NewType,Any
from pathlib import Path
from paddlenlp.trainer import TrainingArguments
from paddlenlp.trainer import PdArgumentParser

DataClass = NewType("DataClass", Any)

@dataclass
class TrainingArguments(TrainingArguments):
    benchmark: bool = field(default=False, metadata={"help": "Whether runs benchmark"})
    profiler_options: Optional[str] = field(default=None, metadata={"help": "Whether runs profiler"})
    warmup_start_lr: float = field(default=1e-6, metadata={"help": "Initial learning rate of warm up."})
    eta_min: float = field(default=1e-5, metadata={"help": "The minimum value of learning rate."})
    lr_scheduler_name: str = field(default="CosineDecayWithWarmup", metadata={"help": "The scheduler name to use."})
    group_by_modality_length: bool = field(default=False)
    mm_projector_lr: Optional[float] = None


@dataclass
class DataArgument:
    dataset: dict = field(default=None, metadata={"help": "config for dataset"})
    task_name: str = field(default=None, metadata={"help": "Additional name to select a more specific task."})
    mixtoken: bool = field(default=False, metadata={"help": "Whether to use MIXToken data stream"})
    src_length: int = field(default=1024, metadata={"help": "The maximum length of source(context) tokens."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When mixtokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    save_generation_output: bool = field(
        default=False,
        metadata={"help": "Whether to save generated text to file when eval_with_do_generation set to True."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )
    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. If is None, it will not use `chat_template.json`; If is equal with `model_name_or_path`, it will use the default loading; If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
        },
    )
    splits: Optional[List[str]] = field(default=None, metadata={"help": "The splits of dataset"})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Built-in pretrained model name or the path to local model."}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    freeze_include: Optional[List[str]] = field(default=None, metadata={"help": "Modules to freeze"})
    freeze_exclude: Optional[List[str]] = field(default=None, metadata={"help": "Modules not to freeze"})

    # LoRA related parameters
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "Lora attention alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora attention dropout"})
    lora_target_modules: List[str] = field(default=None, metadata={"help": "Lora target modules"})

    # prefix tuning related parameters
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    num_prefix_tokens: int = field(default=128, metadata={"help": "Number of prefix tokens"})

    from_aistudio: bool = field(default=False, metadata={"help": "Whether to load model from aistudio"})
    save_to_aistudio: bool = field(default=False, metadata={"help": "Whether to save model to aistudio"})
    aistudio_repo_id: str = field(default=None, metadata={"help": "The id of aistudio repo"})
    aistudio_repo_private: bool = field(default=True, metadata={"help": "Whether to create a private repo"})
    aistudio_repo_license: str = field(default="Apache License 2.0", metadata={"help": "The license of aistudio repo"})
    aistudio_token: str = field(default=None, metadata={"help": "The token of aistudio"})
    neftune: bool = field(default=False, metadata={"help": "Whether to apply NEFT"})
    neftune_noise_alpha: float = field(default=5.0, metadata={"help": "NEFT noise alpha"})
    text_model_name_or_path: str = field(default=None, metadata={"help": "The text tokenizer model name or path"})


@dataclass
class GenerateArgument:
    top_k: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    top_p: float = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )


class PdMIXArgumentParser(PdArgumentParser):
    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        import dataclasses
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
