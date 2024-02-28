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

import argparse

import paddle


def main(args):
    state_dict = paddle.load(args.model_file)
    dtype = args.dtype

    transformer = {}
    vae = {}
    transformer_ema = {}
    for k, v in state_dict.items():
        transformer_key = "transformer."
        if k.startswith(transformer_key):
            transformer[k.replace(transformer_key, "")] = v.astype(dtype)

        transformer_ema_key = "model_ema."
        if k.startswith(transformer_ema_key):
            transformer_ema[k.replace(transformer_ema_key, "")] = v.astype(dtype)

        vae_key = "vae."
        vqvae_key = "vqvae."
        if k.startswith(vae_key):
            vae[k.replace(vae_key, "")] = v.astype(dtype)
        elif k.startswith(vqvae_key):
            vae[k.replace(vqvae_key, "")] = v.astype(dtype)

    if args.save_ema:
        paddle.save(transformer_ema, args.checkpoint_path)
    else:
        paddle.save(transformer, args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        default="",
        type=str,
        required=False,
        help="trained model weight path.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        required=False,
        help="data type.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        type=str,
        required=False,
        help="extracted model weight path.",
    )
    parser.add_argument(
        "--save_ema",
        default=False,
        type=bool,
        required=False,
        help="whether to save ema model.",
    )

    args = parser.parse_args()
    main(args)

# python tools/extract_weights.py --model_file output_trainer/DiT_XL_patch2_trainer/checkpoint-10000/model_state.pdparams --checkpoint_path DiT_XL_patch2_256_global_steps_10000.pdparams
