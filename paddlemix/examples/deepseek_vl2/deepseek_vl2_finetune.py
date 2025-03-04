# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import logging
import math
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import Dataset
from paddlenlp.data import DataCollatorForSeq2Seq  # , DataCollatorWithPadding
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.trainer.trainer import Trainer
from paddlenlp.trainer.trainer_utils import get_last_checkpoint
from paddlenlp.transformers import DeepseekTokenizerFast
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError

from paddlemix.datasets.internvl_dataset import ConcatDataset, WeightedConcatDataset
from paddlemix.models.deepseek_vl2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
from paddlemix.models.qwen2_vl.supervised import _encode_supervised_example
from paddlemix.models.qwen2_vl.template import TEMPLATES
from paddlemix.processors.deepseek_vl2_processing import (
    DeepseekVLChatProcessorOutput,
    DeepseekVLV2Processor,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
logger = logging.getLogger(__name__)


# Set constants for image processing and logging
IGNORE_INDEX = -100
VIDEO_PLACEHOLDER = "<video>"
IMAGE_PLACEHOLDER = "<image>"


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor
    """

    image_resolution: int = field(
        default=384,
        metadata={"help": "Keeps the height or width of image below this resolution."},
    )


@dataclass
class ModelArguments(ProcessorArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the LLM decoder."},
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "Set the drop path rate for the ViT model. Default is 0."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use lora to train model."},
    )
    lora_path: Optional[str] = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: Optional[int] = field(
        default=128,
        metadata={"help": "Set the value of rank in lora. Default is 128."},
    )
    lora_alpha: Optional[int] = field(
        default=256,
        metadata={"help": "Set the value of alpha in lora. Default is 256."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Set the value of dropout in lora. Default is 0.0."},
    )
    lora_target_modules: Optional[str] = field(default=None, metadata={"help": "Lora target modules."})


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_image_size: Optional[int] = field(
        default=384,
        metadata={"help": "Set the desired size for the image. Default is 224."},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={"help": "Pad the image to a square shape if set to True."},
    )
    conv_style: Optional[str] = field(default="deepseek", metadata={"help": "Prompt style for a conversation."})
    meta_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the meta file of datasets."},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use data resampling."},
    )
    normalize_type: Optional[str] = field(
        default="imagenet",
        metadata={"help": "The normalize type for the image. Default is imagenet."},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    group_by_length: bool = field(
        default=True,
        metadata={"help": ""},
    )
    save_safetensors: bool = field(
        default=True,
        metadata={"help": ""},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark (True/False)."},
    )


def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx : idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template,
        meta,
        tokenizer,
        ds_name,
        processor,
        max_image_size=384,
        max_seq_length=2048,
        repeat_time=1,
        normalize_type="imagenet",
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.template = template

        self.processor = processor
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.max_image_size = max_image_size
        self.max_seq_length = max_seq_length
        logger.info("Formatting inputs...Skip in lazy mode")
        if "annotation" in meta:
            meta_anns = meta["annotation"]
        elif "file_name" in meta:
            meta_anns = meta["file_name"]
        else:
            raise ValueError("No annotation found in the meta file.")

        with open(meta_anns, "r") as f:  # qwen2_vl 读的是json
            self.raw_data = json.load(f)
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[: int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        self.cached_data_dict = {}
        self.normalize_type = normalize_type

    def __len__(self):
        return len(self.raw_data)

    def load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def get_image_path(self, image_path):
        # image_path = os.path.join(self.root, image_path)
        return image_path

    def multi_modal_get_item(self, data_item):
        # data_item = {
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": "<image>Using LaTeX to perform OCR on the image."
        #         },
        #         {
        #             "role": "assistant",
        #             "content": "S = \\frac { 2 \\pi R } { n } \\sqrt { E _ { c } ( 2 E - E _ { c } ) } ."
        #         }
        #     ],
        #     "images": [
        #         "LaTeX_OCR/full/train/014484.png"
        #     ]
        # }

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["messages"][0]["content"]:
            data_item["messages"][0]["content"] = "<image>\n" + data_item["messages"][0]["content"]

        # Merge the image path
        image_path = self.get_image_path(data_item["images"][0])  # TODO: now only single image

        messages = data_item["messages"]

        # Load the image using tcs_loader if available, otherwise use PIL
        images = self.load_image(image_path)
        images = [images]

        input_ids, labels = _encode_supervised_example(
            messages=messages,
            system="",
            tools="",
            images=[image_path],
            videos=[],
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            cutoff_len=self.max_seq_length,
            train_on_prompt=False,
            mask_history=False,
        )
        # print('input_ids', input_ids)
        input_ids[0] = 0  # shit

        processor = self.processor
        images_seq_mask = [False] * len(input_ids)
        idx_list = findall(input_ids, processor.image_token_id)  # '<image>'
        _, images_list, _, images_spatial_crop, num_image_tokens = processor.tokenize_with_images(
            "<image>" * len(images), images, cropping=len(images) <= 2
        )
        new_num_tokens = 0
        # print('idx_list', idx_list) # [4]
        # print('input_ids', input_ids)
        # print('num_image_tokens', num_image_tokens)
        # print('1 len(input_ids), len(labels), len(images_seq_mask)', len(input_ids), len(labels), len(images_seq_mask))

        for idx, n_image_tokens in zip(idx_list, num_image_tokens):
            image_tokens = [processor.image_token_id] * n_image_tokens
            input_ids = input_ids[:idx] + image_tokens + input_ids[idx + 1 :]
            if labels is not None:
                labels = labels[:idx] + [-100] * n_image_tokens + labels[idx + 1 :]
            images_seq_mask = images_seq_mask[:idx] + [True] * n_image_tokens + images_seq_mask[idx + 1 :]
            new_num_tokens += n_image_tokens - 1
        # print('2 len(input_ids), len(labels), len(images_seq_mask)', len(input_ids), len(labels), len(images_seq_mask))
        # print('input_ids', input_ids)
        # print('labels', labels)

        output = DeepseekVLChatProcessorOutput(
            sft_format=None,
            input_ids=paddle.to_tensor(input_ids),
            target_ids=paddle.to_tensor(input_ids),
            images=paddle.stack(images_list) if images_list else paddle.zeros((0, 3, 384, 384)),
            images_seq_mask=paddle.to_tensor(images_seq_mask),
            images_spatial_crop=paddle.to_tensor(images_spatial_crop),
            num_image_tokens=num_image_tokens,
        )

        attention_mask = [1] * len(output["input_ids"])

        # Create the final return dictionary
        ret = dict(
            input_ids=output["input_ids"],
            labels=labels,
            attention_mask=attention_mask,
            images=output["images"],
            images_seq_mask=output["images_seq_mask"],
            images_spatial_crop=output["images_spatial_crop"],
        )
        # batched_output = dict(self.processor.batchify(output))
        return ret

    def __getitem__(self, i) -> Dict[str, paddle.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = self.raw_data[i]
                if "images" in data_item and len(data_item["images"]) != 0:
                    ret = self.multi_modal_get_item(data_item)  # TODO: 暂时都是单图
                else:
                    raise NotImplementedError
                    # ret = self.pure_text_get_item(data_item)  # TODO: 纯文
                break
            except Exception as e:
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = self.raw_data[i]
                if "images" in data_item:
                    if type(data_item["images"]) == list:
                        images = [item for item in data_item["images"]]
                        print(f"Failed to load image: {images}, the dataset is: {self.ds_name}")
                    else:
                        data_path = data_item["images"]
                        print(f"Failed to load image: {data_path}, the dataset is: {self.ds_name}")
                else:
                    raise NotImplementedError
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    template,
    tokenizer,
    processor,
    normalize_type="imagenet",
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]["repeat_time"]
        dataset = LazySupervisedDataset(
            template=template,
            meta=ds_collections[ds_name],
            tokenizer=tokenizer,
            ds_name=ds_name,
            processor=processor,
            max_image_size=data_args.max_image_size,
            max_seq_length=data_args.max_seq_length,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
        )
        logger.info(f"Add dataset: {ds_name} with length: {len(dataset)}")
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


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
    # model.image_newline, shape: [1280], requires grad: True
    # model.view_seperator, shape: [1280], requires grad: True
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataTrainingArguments, PreTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters:\n {training_args}")

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
    if "npu" in paddle.get_device():
        is_bfloat16_supported = True
    else:
        is_bfloat16_supported = paddle.amp.is_bfloat16_supported()

    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16 and is_bfloat16_supported:
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

    MODEL_NAME = model_args.model_name_or_path
    model = DeepseekVLV2ForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    tokenizer = DeepseekTokenizerFast.from_pretrained(MODEL_NAME)
    config = DeepseekVLV2Config.from_pretrained(MODEL_NAME)

    candidate_resolutions = config["candidate_resolutions"]
    patch_size = config.vision_config["patch_size"]
    downsample_ratio = config["downsample_ratio"]
    processor = DeepseekVLV2Processor(
        tokenizer=tokenizer,
        candidate_resolutions=candidate_resolutions,
        patch_size=patch_size,
        downsample_ratio=downsample_ratio,
    )

    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    # print("tokenizer", tokenizer)
    # print("len(tokenizer)", len(tokenizer))
    # print("tokenizer.added_tokens_encoder", tokenizer.added_tokens_encoder)
    # print("tokenizer.added_tokens_decoder", tokenizer.added_tokens_decoder)

    data_args.max_image_size = model_args.image_resolution
    train_dataset = build_datasets(
        data_args,
        template=TEMPLATES[data_args.conv_style],
        tokenizer=tokenizer,
        processor=processor,
        normalize_type=data_args.normalize_type,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.stop_gradient = not False

    if model_args.freeze_vit:
        _freeze_params(model.vision)

    if model_args.freeze_llm:
        model.language = model.language.eval()
        _freeze_params(model.language)

    # lora
    if model_args.lora:
        if model_args.lora_path is None:
            target_modules = model_args.lora_target_modules.split(",")  #
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                merge_weights=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    print_trainable_params(model)
    # tiny torch: PeftModelForCausalLM: 3408.4452M Params (37.9438M Trainable [1.1132%]), 0.0008M Buffers.
    # tiny paddle: trainable params: 37943808 || all params: 3408445248 || trainable%: 1.1132
    # small torch : PeftModelForCausalLM: 16290.2329M Params (141.8834M Trainable [0.8710%]), 14.1566M Buffers.
    # small paddle: trainable params: 141883392 || all params: 16290232896 || trainable%: 0.8710

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                logger.info(name)

    # set seed for paddle dataloaders
    set_seed(training_args.seed)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        max_length=4096,
    )

    # data_collator = DataCollatorWithPadding(
    #     tokenizer=tokenizer,
    #     padding=True,
    #     max_length=4096,
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
