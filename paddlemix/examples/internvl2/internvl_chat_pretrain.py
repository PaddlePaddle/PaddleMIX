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

import logging
import os
import sys
import warnings
from functools import partial

import paddle
import paddle.distributed as dist
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import AutoModelForCausalLM, LlamaConfig
from PIL import Image, ImageFile, PngImagePlugin
from train.args_utils import DataTrainingArguments, ModelArguments
from train.dataset import TCSLoader, build_datasets
from train.dataset_packed import packed_collate_fn

from paddlemix.models.internvl2.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from paddlemix.models.internvl2.internlm2 import InternLM2ForCausalLM
from paddlemix.models.internvl2.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
from paddlemix.models.internvl2.patch import concat_pad_data_collator
from paddlemix.models.qwen2_vl.mix_qwen2_tokenizer import MIXQwen2Tokenizer

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError:
    print("petrel_client is not installed. Using PIL to load images.")
    has_tcs_loader = False

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def replace_tokenizer_length(tokenizer):
    def fn(self):
        return len(self.get_vocab())

    tokenizer.__len__ = fn


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / x**0.5
    raise NotImplementedError(loss_reduction)


def main():
    # launcher = os.environ.get('LAUNCHER', 'slurm')
    # init_dist(launcher=launcher, backend='nccl')
    parser = PdArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.use_packed_ds = data_args.use_packed_ds
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, _n_gpu: {dist.get_world_size()}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    if "qwen" in tokenizer_path.lower():
        tokenizer = MIXQwen2Tokenizer.from_pretrained(tokenizer_path, add_eos_token=False, trust_remote_code=True)

        # tokenizer.added_tokens_encoder =  {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645, '<img>': 151646, '</img>': 151647, '<IMG_CONTEXT>': 151648, '<quad>': 151649, '</quad>': 151650, '<ref>': 151651, '</ref>': 151652, '<box>': 151653, '</box>': 151654}
        # tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}
    else:
        from paddlemix.models.internvl2.internlm2 import InternLM2Tokenizer

        tokenizer = InternLM2Tokenizer.from_pretrained(tokenizer_path, add_eos_token=False, trust_remote_code=True)
    replace_tokenizer_length(tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    add_start_idx = len(tokenizer)
    for idx, new_token in enumerate(token_list):
        tokenizer.added_tokens_encoder[new_token] = add_start_idx + idx
    tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader("~/petreloss.conf") if has_tcs_loader else None

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
    if model_args.model_name_or_path is not None:
        logger.info("Loading InternVLChatModel...")
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for InternLM")
        else:
            config.llm_config._attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for LLaMA")
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(model_args.model_name_or_path, dtype=dtype, config=config)
    else:
        logger.info("Loading ViT-6B...")
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(model_args.vision_path, dtype=dtype, config=vision_config)
        logger.info("Loading LLaMA...")
        llm_config = LlamaConfig.from_pretrained(model_args.llm_path)
        if llm_config.model_type == "internlm2":
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for InternLM")
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for LLaMA")
        llm = model_type.from_pretrained(model_args.llm_path, dtype=dtype, config=llm_config)
        logger.info("Building InternVLChatConfig...")
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
        )
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info("Building InternVLChatModel...")
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    assert model.config.downsample_ratio == data_args.down_sample_ratio
    if model_args.mlp_path is not None:
        logger.info("Loading pretrained MLP projector...")
        state_dict = paddle.load(path=str(model_args.mlp_path))
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info("Finished")
    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(f"model.config.vision_config.image_size: {model.config.vision_config.image_size}")
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f"Resizing position embedding from {model.config.vision_config.image_size} to {data_args.force_image_size}..."
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size, new_size=data_args.force_image_size, patch_size=patch_size
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * data_args.down_sample_ratio**2)

    if num_new_tokens > 0:
        model.language_model.config.tie_word_embeddings = True  # resize lm_head to new vocab size
        model.language_model.resize_token_embeddings(len(tokenizer))
        # tensor with gradient can not inplace
        with paddle.no_grad():
            output_embeddings = model.language_model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(axis=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    model.language_model.config.use_cache = False

    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=False,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.stop_gradient = not False

    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)
    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)
    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.stop_gradient = not True

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)
    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.stop_gradient = not True
    if paddle.distributed.get_rank() == 0:
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                logger.info(name)
    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
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
