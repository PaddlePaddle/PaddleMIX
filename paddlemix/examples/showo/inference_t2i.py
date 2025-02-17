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

import numpy as np
import paddle
from omegaconf import OmegaConf
from paddlenlp.transformers import CodeGenTokenizer
from PIL import Image
from tqdm import tqdm

from paddlemix.models.showo import MAGVITv2, Showo, get_mask_chedule
from paddlemix.models.showo.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


if __name__ == "__main__":

    config = get_config()

    tokenizer = CodeGenTokenizer.from_pretrained(config.model.showo.llm_model_path)

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.eval()

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path, dtype=config.dtype)
    model.eval()

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    # load from users passed arguments

    if not os.path.exists("./gen_res"):
        os.mkdir("./gen_res")

    if config.mode == "t2i":
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            validation_prompts = f.read().splitlines()

        for step in tqdm(range(0, len(validation_prompts), config.training.batch_size)):
            prompts = validation_prompts[step : step + config.training.batch_size]

            image_tokens = (
                paddle.ones((len(prompts), config.model.showo.num_vq_tokens), dtype=paddle.int64) * mask_token_id
            )

            input_ids, _ = uni_prompting((prompts, image_tokens), "t2i_gen")
            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([""] * len(prompts), image_tokens), "t2i_gen")
                attention_mask = create_attention_mask_predict_next(
                    paddle.concat([input_ids, uncond_input_ids], axis=0),
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
            else:
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
            # import pdb;pdb.set_trace()
            with paddle.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,  # [2, 1, 1155, 1155]
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
            print("gen_token_ids", gen_token_ids)

            # gen_token_ids = paddle.to_tensor(np.load('gen_token_ids.npy'), dtype=paddle.int64)
            # [[5202, 4707, 327 , ..., 2239, 3625, 2237]]
            gen_token_ids = paddle.clip(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            # gen_token_ids [1, 1024]
            images = vq_model.decode_code(gen_token_ids)

            images = paddle.clip((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]
            print(f"save at: gen_res/image_{step}.jpg")
            pil_images[0].save(f"gen_res/image_{step}.jpg")
