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

DEFAULT = {
    "model": "PPDocBee-2B-1129",
    "lang": "zh",
    "model_path": "PaddleMIX/PPDocBee-2B-1129",
}

SUPPORTED_MODELS = {
    "PPDocBee-2B-1129": "PaddleMIX/PPDocBee-2B-1129",
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
}


MODEL_MAPPING = {
    "PPDocBee-2B-1129": "Qwen2VLForConditionalGeneration",
    "Qwen2-VL-2B-Instruct": "Qwen2VLForConditionalGeneration",
    "Qwen2-VL-7B-Instruct": "Qwen2VLForConditionalGeneration",
    "Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLForConditionalGeneration",
    "Qwen2.5-VL-7B-Instruct": "Qwen2_5_VLForConditionalGeneration",
    "Qwen2.5-VL-72B-Instruct": "Qwen2_5_VLForConditionalGeneration",
}

TOKENIZER_MAPPING = {
    "PPDocBee-2B-1129": "MIXQwen2Tokenizer",
    "Qwen2-VL-2B-Instruct": "MIXQwen2Tokenizer",
    "Qwen2-VL-7B-Instruct": "MIXQwen2Tokenizer",
    "Qwen2.5-VL-3B-Instruct": "MIXQwen2_5_Tokenizer",
    "Qwen2.5-VL-7B-Instruct": "MIXQwen2_5_Tokenizer",
    "Qwen2.5-VL-72B-Instruct": "MIXQwen2_5_Tokenizer",
}

VL_PROCESSOR_MAPPING = {
    "PPDocBee-2B-1129": "Qwen2VLProcessor",
    "Qwen2-VL-2B-Instruct": "Qwen2VLProcessor",
    "Qwen2-VL-7B-Instruct": "Qwen2VLProcessor",
    "Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLProcessor",
    "Qwen2.5-VL-7B-Instruct": "Qwen2_5_VLProcessor",
    "Qwen2.5-VL-72B-Instruct": "Qwen2_5_VLProcessor",
}

IMAGE_PROCESSOR_MAPPING = {
    "PPDocBee-2B-1129": "Qwen2VLImageProcessor",
    "Qwen2-VL-2B-Instruct": "Qwen2VLImageProcessor",
    "Qwen2-VL-7B-Instruct": "Qwen2VLImageProcessor",
    "Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLImageProcessor",
    "Qwen2.5-VL-7B-Instruct": "Qwen2_5_VLImageProcessor",
    "Qwen2.5-VL-72B-Instruct": "Qwen2_5_VLImageProcessor",
}

VISION_PROCESS_MAPPING = {
    "PPDocBee-2B-1129": "qwen2_vl_processing",
    "Qwen2-VL-2B-Instruct": "qwen2_vl_processing",
    "Qwen2-VL-7B-Instruct": "qwen2_vl_processing",
    "Qwen2.5-VL-3B-Instruct": "qwen2_5_vl_processing",
    "Qwen2.5-VL-7B-Instruct": "qwen2_5_vl_processing",
    "Qwen2.5-VL-72B-Instruct": "qwen2_5_vl_processing",
}

DEFAULT_TEMPLATE = {
    "default": "qwen2_vl",
    "PPDocBee-2B-1129": "qwen2_vl",
    "Qwen2-VL-2B-Instruct": "qwen2_vl",
    "Qwen2-VL-7B-Instruct": "qwen2_vl",
    "Qwen2.5-VL-3B-Instruct": "qwen2_5_vl",
    "Qwen2.5-VL-7B-Instruct": "qwen2_5_vl",
    "Qwen2.5-VL-72B-Instruct": "qwen2_5_vl",
}

TRAINABLE_MODELS = {
    "sft": ["PPDocBee-2B-1129", "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"],
}

METHODS = ["full", "lora"]

# train
TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
}

STAGES_USE_PAIR_DATA = {}
PADDLEMIX_CONFIG = "config.yaml"

DATA_CONFIG = "dataset_info.json"

PEFT_METHODS = {"lora"}

DEFAULT_DATA_DIR = "data"

TRAINER_MAPPING = {}

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.environ.get("VIDEO_PLACEHOLDER", "<video>")

CHECKPOINT_NAMES = {}

FINAL_CHECKPOINT_NAME = "checkpoint-latest"

TRAINER_LOG = "trainer_log.jsonl"
RUNNING_LOG = "running_log.txt"
