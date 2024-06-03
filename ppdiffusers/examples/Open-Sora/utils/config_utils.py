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


def parse_args():
    parser = argparse.ArgumentParser()

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=42, type=int, help="generation seed")

    parser.add_argument("--batch-size", default=1, type=int, help="batch size")

    # ======================================================
    # Inference
    # ======================================================
    parser.add_argument("--save-dir", default="./samples/samples/", type=str, help="path to save generated samples")
    parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
    parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
    parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
    parser.add_argument("--num-sample", default=1, type=int, help="number of samples to generate for one prompt")
    parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")

    # prompt
    parser.add_argument(
        "--prompt-path", default="./assets/texts/t2v_samples.txt", type=str, help="path to prompt txt file"
    )
    parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")

    # image/video
    parser.add_argument("--num-frames", default=12, type=int, help="number of frames")
    parser.add_argument("--fps", default=24, type=int, help="fps")
    parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")

    # reference
    parser.add_argument("--loop", default=1, type=int, help="loop")
    parser.add_argument("--condition-frame-length", default=None, type=int, help="condition frame length")
    parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
    parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")

    # args from config sample.py

    parser.add_argument("--frame_interval", default=3, type=int, help="frame_interval")

    parser.add_argument("--multi_resolution", default="STDiT2", type=str, help="multi_resolution")
    parser.add_argument("--dtype", default="fp32", type=str, help="dtype_infernece")

    # args for module definition
    parser.add_argument(
        "--vae_pretrained_path", default="stabilityai/sd-vae-ft-ema", type=str, help="vae_pretrained_path"
    )
    parser.add_argument("--micro_batch_size", default=1, type=int, help="micro_batch_size for vae")

    parser.add_argument(
        "--text_encoder_pretrained_path",
        default="DeepFloyd/t5-v1_1-xxl",
        type=str,
        help="text_encoder_pretrained_path",
    )
    parser.add_argument("--model_max_length", default=200, type=int, help="model_max_length for text_encoder")

    parser.add_argument(
        "--model_pretrained_path",
        default="hpcai-tech/OpenSora-STDiT-v2-stage3",
        type=str,
        help="model_pretrained_path",
    )

    parser.add_argument("--num_sampling_steps", default=100, type=int, help="num_sampling_steps for scheduler")
    parser.add_argument("--cfg-scale", default=7.0, type=float, help="balance between cond & uncond")
    parser.add_argument("--cfg_channel", default=3, type=int, help="cfg_channel for scheduler")

    return parser.parse_args()


def parse_configs():
    args = parse_args()

    return args
