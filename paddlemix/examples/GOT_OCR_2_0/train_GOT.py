# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import paddle
import paddle.distributed as dist
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.trainer.trainer import Trainer
from paddlenlp.trainer.trainer_utils import get_last_checkpoint
from paddlenlp.transformers import QWenTokenizer

from paddlemix.datasets.got_dataset import make_supervised_data_module
from paddlemix.models.GOT.GOT_ocr_2_0 import GOTQwenForCausalLM
from paddlemix.models.GOT.utils.utils import smart_tokenizer_and_embedding_resize

logger = logging.getLogger(__name__)


def print_trainable_params(model: paddle.nn.Layer) -> None:
    trainable_params, all_param = 0, 0
    for k, param in model.named_parameters():
        num_params = param.size
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if not param.stop_gradient:
            # print('{}, shape: {}, requires grad: {}'.format(k, param.shape, not param.stop_gradient))
            trainable_params += num_params
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="stepfun-ai/GOT-OCR2_0")
    use_cache: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14")
    freeze_vision_tower: bool = field(default=False)
    freeze_lm_model: bool = field(default=False)
    pretrained_stage1_model: Optional[str] = field(default=None)  # mlp &/ vision tower
    vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    use_im_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    datasets: str = field(default=None, metadata={"help": "combinations of the training data."})
    meta_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the meta file of datasets."},
    )
    sep_image_conv_front: bool = False
    image_token_len: int = 256
    image_aspect_ratio: str = "square"
    conversation_version: str = "mpt"
    box_limit: int = 0
    max_seq_length: int = 8192


@dataclass
class GOTTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    interleave: bool = field(default=False)
    with_box: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def train():
    parser = PdArgumentParser((ModelArguments, DataArguments, GOTTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Detecting last checkpoint and eventually continue from last checkpoint.
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

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16 and paddle.amp.is_bfloat16_supported():
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path
    print(f"Loading Tokenizer: {tokenizer_path}")

    tokenizer = QWenTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="right", model_max_length=training_args.model_max_length
    )
    print("tokenizer", tokenizer)
    # print("len(tokenizer)", len(tokenizer))
    # print("tokenizer.added_tokens_encoder", tokenizer.added_tokens_encoder)
    # print("tokenizer.added_tokens_decoder", tokenizer.added_tokens_decoder)

    model = GOTQwenForCausalLM.from_pretrained(model_args.model_name_or_path, dtype=dtype)

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="<|endoftext|>"),
        tokenizer=tokenizer,
        model=model,
    )

    vision_tower_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=model_args.freeze_vision_tower,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
    )

    model.initialize_vision_tokenizer(
        tokenizer=tokenizer,
        freeze_lm_model=model_args.freeze_lm_model,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
    )

    # 'image_processor_high
    data_args.image_token_len = 256
    data_args.image_processor = vision_tower_dict["image_processor"]
    data_args.image_processor_high = vision_tower_dict["image_processor_high"]
    data_args.use_im_start_end = model_args.use_im_start_end

    def _freeze_params(module):
        for param in module.parameters():
            param.stop_gradient = not False

    # mixed relation, to be fixed
    if model_args.freeze_lm_model:
        _freeze_params(model.get_model().mm_projector)
        _freeze_params(model.get_model().mm_projector_vary)
        _freeze_params(model.get_input_embeddings())

    if model_args.freeze_vision_tower:
        _freeze_params(model.qwen2.vision_tower_high)

    print_trainable_params(model)
    # trainable params: 464959488 || all params: 560528640 || trainable%: 82.9502 # stage3
    # trainable params: 560528640 || all params: 560528640 || trainable%: 100 # stage2
    params_grad = [p.numel() for n, p in model.named_parameters() if not p.stop_gradient]
    print(f"Number of Mapping Trainable Parameters: {int(sum(params_grad)) / (1 << 20):.2f} M")

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                logger.info(name)

    # set seed for paddle dataloaders
    set_seed(training_args.seed)

    data_module = make_supervised_data_module(
        interleave=training_args.interleave, with_box=training_args.with_box, tokenizer=tokenizer, data_args=data_args
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        **data_module,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(data_module["train_dataset"])
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    train()
