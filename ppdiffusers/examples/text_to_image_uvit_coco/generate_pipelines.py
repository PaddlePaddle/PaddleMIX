# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import json

import paddle
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LDMTextToImagePipeline,
    UViTModel_T2I,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./model_state.pdparams",
        help="path to pretrained model_state.pdparams",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./ldm_uvit_pipelines",
        help="the output path of pipeline.",
    )
    parser.add_argument(
        "--unet_config_file",
        type=str,
        default="examples/text_to_image_laion400m/config/uvit_t2i_small_deep.json",
        help="unet_config_file.",
    )
    parser.add_argument(
        "--vae_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4/vae",
        help="pretrained_vae_name_or_path.",
    )
    parser.add_argument(
        "--text_encoder_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5/text_encoder",
        help="Pretrained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5/tokenizer",
        help="Pretrained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--scheduler_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5/scheduler",
        help="Pretrained scheduler name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=77,
        help="Pretrained tokenizer model_max_length.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like gpu:0 or cpu")
    parser.add_argument("--use_ema", type=bool, default=True, help="Use ema model")

    return parser.parse_args()


def extract_paramaters(model_file="model_state.pdparams", use_ema=True, dtype="float32"):
    state_dict = paddle.load(model_file)
    unet = {}
    vae = {}
    unet_ema = {}
    for k, v in state_dict.items():
        unet_key = "unet."
        if k.startswith(unet_key):
            unet[k.replace(unet_key, "")] = v.astype(dtype)

        unet_key = "model_ema."
        if k.startswith(unet_key):
            unet_ema[k.replace(unet_key, "")] = v.astype(dtype)

        vae_key = "vae."
        vqvae_key = "vqvae."
        if k.startswith(vae_key):
            vae[k.replace(vae_key, "")] = v.astype(dtype)
        elif k.startswith(vqvae_key):
            vae[k.replace(vqvae_key, "")] = v.astype(dtype)

    if use_ema and len(unet_ema) > 0:
        new_unet_ema = {}
        for k, v in unet.items():
            flat_ema_key = "".join(k.split("."))
            new_unet_ema[k] = unet_ema[flat_ema_key]
        return new_unet_ema, vae
    else:
        return unet, vae


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def check_keys(model, state_dict):
    cls_name = model.__class__.__name__
    missing_keys = []
    mismatched_keys = []
    for k, v in model.state_dict().items():
        if k not in state_dict.keys():
            missing_keys.append(k)
        if list(v.shape) != list(state_dict[k].shape):
            mismatched_keys.append(k)
    if len(missing_keys):
        missing_keys_str = ", ".join(missing_keys)
        print(f"{cls_name} Found missing_keys {missing_keys_str}!")
    if len(mismatched_keys):
        mismatched_keys_str = ", ".join(mismatched_keys)
        print(f"{cls_name} Found mismatched_keys {mismatched_keys_str}!")


def build_pipelines(
    model_file,
    output_path,
    vae_name_or_path,
    unet_config_file,
    text_encoder_name_or_path,
    tokenizer_name_or_path,
    scheduler_name_or_path,
    model_max_length=77,
    use_ema=True,
):
    vae = AutoencoderKL.from_config(vae_name_or_path)
    unet = UViTModel_T2I(**read_json(unet_config_file))
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=model_max_length)
    scheduler = DDIMScheduler.from_pretrained(scheduler_name_or_path)

    unet_dict, vae_dict = extract_paramaters(model_file, use_ema)
    check_keys(unet, unet_dict)
    check_keys(vae, vae_dict)
    unet.load_dict(unet_dict)
    vae.load_dict(vae_dict)

    pipe = LDMTextToImagePipeline(
        bert=text_encoder,  # still named bert
        tokenizer=tokenizer,
        scheduler=scheduler,
        vqvae=vae,  # still named vqvae
        unet=unet,
    )
    pipe.save_pretrained(output_path)


if __name__ == "__main__":
    args = parse_args()
    if args.device is not None:
        paddle.set_device(args.device)
    build_pipelines(
        model_file=args.model_file,
        output_path=args.output_path,
        vae_name_or_path=args.vae_name_or_path,
        unet_config_file=args.unet_config_file,
        text_encoder_name_or_path=args.text_encoder_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        scheduler_name_or_path=args.scheduler_name_or_path,
        model_max_length=args.model_max_length,
        use_ema=args.use_ema,
    )
