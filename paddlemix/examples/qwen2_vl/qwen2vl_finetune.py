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

import json
import math
import time
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Any

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import Dataset
from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.utils import profiler
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.trainer.trainer import PrinterCallback, ProgressCallback, Trainer
from paddlenlp.trainer.integrations import TrainerCallback
from paddlenlp.trainer.trainer_utils import get_last_checkpoint
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError

from paddlemix.datasets.internvl_dataset import ConcatDataset, WeightedConcatDataset
from paddlemix.models.qwen2_vl import MIXQwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.models.qwen2_vl.supervised import _encode_supervised_example
from paddlemix.models.qwen2_vl.template import TEMPLATES
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
)
from paddlenlp.transformers.processing_utils import ProcessorMixin

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
from paddlenlp.utils.log import logger

# Set constants for image processing and logging
IGNORE_INDEX = -100
VIDEO_PLACEHOLDER = "<video>"
IMAGE_PLACEHOLDER = "<image>"


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """

    image_resolution: int = field(
        default=512,
        metadata={"help": "Keeps the height or width of image below this resolution."},
    )
    video_resolution: int = field(
        default=128,
        metadata={"help": "Keeps the height or width of video below this resolution."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=64,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
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
        default=8192,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_image_size: Optional[int] = field(
        default=512,
        metadata={"help": "Set the desired size for the image. Default is 224."},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={"help": "Pad the image to a square shape if set to True."},
    )
    conv_style: Optional[str] = field(default="qwen2_vl", metadata={"help": "Prompt style for a conversation."})
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
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "Whether runs profiler"},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template,
        meta,
        tokenizer,
        ds_name,
        processor,
        max_image_size=512,
        max_seq_length=8192,
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

    def _preprocess_image(self, image):
        r"""
        Pre-processes a single image.
        """
        image_resolution = self.max_image_size
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self._preprocess_image(image)

    def get_image_path(self, image_path):
        # image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        return self.processor.image_processor

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["messages"][0]["content"]:
            data_item["messages"][0]["content"] = "<image>\n" + data_item["messages"][0]["content"]

        # Merge the image path
        image_path = self.get_image_path(data_item["images"][0])  # TODO: now only single image

        messages = data_item["messages"]

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
        attention_mask = [1] * len(input_ids)

        # Create the final return dictionary
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=[image_path],
        )
        return ret

    def pure_text_get_item(self, data_item):
        messages = data_item["messages"]

        input_ids, labels = _encode_supervised_example(
            messages=messages,
            system="",
            tools="",
            images=[],
            videos=[],
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            cutoff_len=self.max_seq_length,
            train_on_prompt=False,
            mask_history=False,
        )
        attention_mask = [1] * len(input_ids)

        # Create the final return dictionary
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=[],
        )
        
        return ret

    def __getitem__(self, i) -> Dict[str, paddle.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = self.raw_data[i]
                if "images" in data_item and len(data_item["images"]) != 0:
                    ret = self.multi_modal_get_item(data_item)  # TODO: 暂时都是单图
                else:
                    ret = self.pure_text_get_item(data_item)  # TODO: 纯文
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
                elif "video" in data_item:
                    data_path = data_item["video"]
                    print(f"Failed to load video: {data_path}, the dataset is: {self.ds_name}")
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
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images and videos.
    """

    template: Optional["TEMPLATES"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "paddle.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids = [], [], [], [], []

        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_input_ids.append(feature["input_ids"])                

        if (
            self.processor is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(fake_messages, fake_images, [], self.processor)
            fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                fake_input_ids, None, fake_images, [], self.tokenizer, self.processor
            )

            if len(fake_input_ids) != 0:
                if self.tokenizer.padding_side == "right":
                    features[0]["input_ids"] = features[0]["input_ids"]+ fake_input_ids["input_ids"]
                    features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids["input_ids"])
                    features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids["input_ids"])
                else:
                    features[0]["input_ids"] = fake_input_ids["input_ids"] + features[0]["input_ids"]
                    features[0]["attention_mask"] = [0] * len(fake_input_ids["input_ids"]) + features[0]["attention_mask"]
                    features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids["input_ids"]) + features[0]["labels"]

            batch_images = fake_images
            batch_imglens[0] = 1
            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids, self.processor
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: Dict[str, "paddle.Tensor"] = super().__call__(features)

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(
                input_ids=features["input_ids"],
                image_grid_thw=mm_inputs.get("image_grid_thw", None),
                video_grid_thw=mm_inputs.get("video_grid_thw", None),
                attention_mask=features["attention_mask"],
            )

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)
        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # use default_collate() instead of BatchEncoding.to()

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = paddle.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


class AverageStatistical(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cnt = 0
        self.time = 0

    def record(self, val, cnt=1):
        self.time += val
        self.total_cnt += cnt

    def get_average(self):
        if self.total_cnt == 0:
            return 0

        return self.time / self.total_cnt

    def get_average_per_sec(self):
        if self.time == 0.0:
            return 0.0

        return float(self.total_cnt) / self.time

    def get_total_cnt(self):
        return self.total_cnt

    def get_total_time(self):
        return self.time


class BenchmarkCallback(TrainerCallback):
    def __init__(self, benchmark=True, profiler_options=None):
        self.benchmark = benchmark
        self.profiler_options = profiler_options

    def on_train_begin(self, args, state, control, **kwargs):
        # assert args.gradient_accumulation_steps == 1 and not args.do_eval and not args.do_predict
        if self.benchmark:
            self.reader_cost_avg = AverageStatistical()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.epoch_start = time.time()
            self.batch_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.reader_cost_avg.record(time.time() - self.batch_start)

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler_options is not None:
            profiler.add_profiler_step(self.profiler_options)

        if self.benchmark:
            self.batch_start = time.time()
            if control.should_log:
                self.maybe_log_save_evaluate_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_steps_per_second" in logs:
                self.batch_start = self.batch_start + (time.time() - self.maybe_log_save_evaluate_start)
                ips = logs["interval_steps_per_second"] * args.train_batch_size
                avg_batch_cost = 1 / logs["interval_steps_per_second"]
                max_mem_reserved_msg = ""
                max_mem_allocated_msg = ""
                if paddle.device.is_compiled_with_cuda():
                    max_mem_reserved_msg = (
                        f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                    )
                    max_mem_allocated_msg = (
                        f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                    )
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec, %s %s"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
                        max_mem_reserved_msg,
                        max_mem_allocated_msg,
                    )
                )
                self.reader_cost_avg.reset()


class Qwen2VLTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.args.benchmark or self.args.profiler_options is not None:
            self.add_callback(
                BenchmarkCallback(
                    benchmark=self.args.benchmark,
                    profiler_options=self.args.profiler_options,
                )
            )
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)


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
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=dtype)
    image_processor = Qwen2VLImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = MIXQwen2Tokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    processor = Qwen2VLProcessor(image_processor, tokenizer)

    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    print("tokenizer", tokenizer)
    print("len(tokenizer)", len(tokenizer))
    print("tokenizer.added_tokens_encoder", tokenizer.added_tokens_encoder)
    print("tokenizer.added_tokens_decoder", tokenizer.added_tokens_decoder)

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
        _freeze_params(model.visual)

    if model_args.freeze_llm:
        model.model = model.model.eval()
        model.lm_head = model.lm_head.eval()
        _freeze_params(model.model)
        _freeze_params(model.lm_head)

    # lora
    if model_args.lora:
        if model_args.lora_path is None:
            target_modules = model_args.lora_target_modules.split(",")
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

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                logger.info(name)

    # set seed for paddle dataloaders
    set_seed(training_args.seed)

    data_collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        template=TEMPLATES[data_args.conv_style],
        processor=processor,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX,
    )

    trainer = Qwen2VLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    total_samples = len(train_dataset)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if training_args.benchmark:
            def get_paddle_memory_info():
                """get_memory_info"""
                divisor = 2**30
                return (
                    paddle.device.cuda.memory_allocated() / divisor,
                    paddle.device.cuda.max_memory_allocated() / divisor,
                    paddle.device.cuda.memory_reserved() / divisor,
                    paddle.device.cuda.max_memory_reserved() / divisor,
                )
            memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved = get_paddle_memory_info()

            logger.info(f'memory_allocated:{memory_allocated}GB, max_memory_allocated: {max_memory_allocated}GB, memory_reserved:{memory_reserved}GB, max_memory_reserved: {max_memory_reserved}GB \n')
            total_effective_samples = total_samples * training_args.num_train_epochs
            effective_samples_per_second = total_effective_samples / train_result.metrics["train_runtime"]
            # mem_gpu = (
            #     train_result.metrics["train_mem_gpu_peaked_delta"] + train_result.metrics["train_mem_gpu_alloc_delta"]
            # )
            logger.info(f"ips: {effective_samples_per_second} ")
            # logger.info(f"train_mem_gpu_peaked: {int(mem_gpu/ (2**20))} MB")
            logger.info("Benchmark done.")
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()


if __name__ == "__main__":
    main()
