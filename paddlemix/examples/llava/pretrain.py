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
import sys

import paddle
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint
from paddlenlp.transformers import CLIPImageProcessor
from paddlenlp.utils.log import logger

from paddlemix.datasets import MixDataset, MIXTokenMapDataset
from paddlemix.models.llava.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)
from paddlemix.models.llava.language_model.tokenizer import LLavaTokenizer
from paddlemix.processors import LlavaProcessor
from paddlemix.trainer import (
    DataArgument,
    GenerateArgument,
    ModelArgument,
    TrainingArguments,
    freeze_params,
    get_trainer,
)


def main():
    # Arguments
    parser = PdArgumentParser((GenerateArgument, ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        gen_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        gen_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(gen_args, "Generation")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
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

    # Load model config
    model_config = LlavaConfig.from_pretrained(model_args.model_name_or_path, dtype=dtype)
    model_config.use_flash_attention = model_args.use_flash_attention

    # Load model
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        dtype=dtype,
    )

    # Freeze module
    if model_args.freeze_include or model_args.freeze_exclude:
        freeze_params(model, include=model_args.freeze_include, exclude=model_args.freeze_exclude)

    tokenizer = LLavaTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=data_args.max_length)

    # Load processor
    name_or_path = os.path.join(model_args.model_name_or_path, "processor", "train")
    image_processor = CLIPImageProcessor.from_pretrained(name_or_path)
    # Load processor
    train_processor = LlavaProcessor(
        image_processor,
        tokenizer,
        max_length=data_args.max_length,
        version=model_config.version,
    )
    if training_args.do_eval:
        name_or_path = os.path.join(model_args.model_name_or_path, "processor", "eval")
        image_processor = CLIPImageProcessor.from_pretrained(name_or_path)
        eval_processor = LlavaProcessor(
            image_processor,
            tokenizer,
            max_length=data_args.max_length,
            version=model_config.version,
        )

    # Load dataset
    train_ds = None
    eval_ds = None
    if data_args.dataset is None:
        raise ValueError(f"Please specific dataset config (got {data_args.dataset})")
    else:
        if "train" in data_args.dataset.keys():
            train_ds = MixDataset(data_args.dataset["train"])
        if "eval" in data_args.dataset.keys():
            eval_ds = MixDataset(data_args.dataset["eval"])

    total_samples = len(train_ds) if train_ds is not None else 0

    if data_args.mixtoken:
        if (
            model.base_model_prefix not in ["qwen", "visualglm", "llava"]
            and training_args.pipeline_parallel_degree < 1
        ):
            raise NotImplementedError("MIXToke data stream is only implemented for QWen-VL Visualglm llava so far.")
        if model.base_model_prefix == "llava":
            tokenizer.image_token_span = model.llama.vision_tower.num_patches
            logger.info("tokenizer image span: {}".format(tokenizer.image_token_span))
        mixtoken_dataset = MIXTokenMapDataset
        logger.info("Creating MIXToken Data Stream. This may take a few minutes.")
        train_ds = mixtoken_dataset(
            train_ds, max_length=data_args.max_length, processor=train_processor, tokenizer=tokenizer
        )

    # get Trainer
    trainer = get_trainer(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        train_processor=train_processor,
        eval_processor=eval_processor if training_args.do_eval else None,
        mixtokens=data_args.mixtoken,
    )

    # Train
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

            logger.info(
                f"memory_allocated:{memory_allocated}GB, max_memory_allocated: {max_memory_allocated}GB, memory_reserved:{memory_reserved}GB, max_memory_reserved: {max_memory_reserved}GB \n"
            )
            total_effective_samples = total_samples * training_args.num_train_epochs
            effective_samples_per_second = total_effective_samples / train_result.metrics["train_runtime"]
            mem_gpu = (
                train_result.metrics["train_mem_gpu_peaked_delta"] + train_result.metrics["train_mem_gpu_alloc_delta"]
            )
            logger.info(f"ips: {effective_samples_per_second} ")
            logger.info(f"train_mem_gpu_peaked: {int(mem_gpu/ (2**20))} MB")
            logger.info("Benchmark done.")
        else:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()


if __name__ == "__main__":
    main()
