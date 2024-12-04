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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from paddlenlp.trainer import PdArgumentParser

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 4)))
sys.path.insert(0, parent_path)
from paddlemix.models.diffsinger.inference.ds_acoustic import DiffSingerAcousticInfer
from paddlemix.models.diffsinger.utils.hparams import hparams, set_hparams
from paddlemix.models.diffsinger.utils.infer_utils import (
    parse_commandline_spk_mix,
    trans_key,
)

root_dir = Path(__file__).resolve().parent.parent
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))


@dataclass
class DiffsingerArguments:
    """
    Arguments for DiffSinger inference
    """

    # Input/Output arguments
    proj: str = field(metadata={"help": "Path to the DS_FILE for inference"})
    exp: str = field(metadata={"help": "Selection of model"})
    out: Optional[str] = field(default="./output", metadata={"help": "Path of the output folder"})
    title: Optional[str] = field(default=None, metadata={"help": "Title of output file"})

    # Model arguments
    ckpt: Optional[int] = field(default=None, metadata={"help": "Selection of checkpoint training steps"})
    spk: Optional[str] = field(default=None, metadata={"help": "Speaker name or mixture of speakers"})
    num: int = field(default=1, metadata={"help": "Number of runs"})
    key: int = field(default=0, metadata={"help": "Key transition of pitch"})
    gender: Optional[float] = field(default=None, metadata={"help": "Formant shifting (gender control)"})
    seed: int = field(default=-1, metadata={"help": "Random seed of the inference"})
    depth: Optional[float] = field(default=None, metadata={"help": "Shallow diffusion depth"})
    steps: Optional[int] = field(default=None, metadata={"help": "Diffusion sampling steps"})
    mel: bool = field(default=False, metadata={"help": "Save intermediate mel format instead of waveform"})


def find_exp(exp: str) -> str:
    """Find ckpt file in the specified directory"""
    exp_path = Path(exp)

    if not exp_path.exists():
        raise ValueError(f"The specified path '{exp}' does not exist.")

    # Find ckpt file in the directory
    ckpt_files = list(exp_path.glob("*.ckpt"))
    if not ckpt_files:
        raise ValueError(f"No .ckpt file found in directory '{exp}'")
    if len(ckpt_files) > 1:
        print(f"| Warning: Multiple ckpt files found, using the first one: {ckpt_files[0]}")

    print(f"| found ckpt file: {ckpt_files[0]}")
    return exp


def process_hparams(hparams: dict, args: DiffsingerArguments, params: List[dict]) -> Tuple[dict, List[dict]]:
    """Process and update hyperparameters"""
    # Validate vocoder checkpoint
    assert (
        args.mel or Path(hparams["vocoder_ckpt"]).exists()
    ), f"Vocoder ckpt '{hparams['vocoder_ckpt']}' not found. Please put it to the checkpoints directory to run inference."

    # Update speedup parameters
    if "diff_speedup" not in hparams and "pndm_speedup" in hparams:
        hparams["diff_speedup"] = hparams["pndm_speedup"]

    # Set default parameters if not present
    if "T_start" not in hparams:
        hparams["T_start"] = 1 - hparams["K_step"] / hparams["timesteps"]

    if "T_start_infer" not in hparams:
        hparams["T_start_infer"] = 1 - hparams["K_step_infer"] / hparams["timesteps"]

    if "sampling_steps" not in hparams:
        if hparams["use_shallow_diffusion"]:
            hparams["sampling_steps"] = hparams["K_step_infer"] // hparams["diff_speedup"]
        else:
            hparams["sampling_steps"] = hparams["timesteps"] // hparams["diff_speedup"]

    if "time_scale_factor" not in hparams:
        hparams["time_scale_factor"] = hparams["timesteps"]

    # Process depth parameter
    if args.depth is not None:
        assert (
            args.depth <= 1 - hparams["T_start"]
        ), f"Depth should not be larger than 1 - T_start ({1 - hparams['T_start']})"
        hparams["K_step_infer"] = round(hparams["timesteps"] * args.depth)
        hparams["T_start_infer"] = 1 - args.depth

    # Process steps parameter
    if args.steps is not None:
        if hparams["use_shallow_diffusion"]:
            step_size = (1 - hparams["T_start_infer"]) / args.steps
            if "K_step_infer" in hparams:
                hparams["diff_speedup"] = round(step_size * hparams["K_step_infer"])
        elif "timesteps" in hparams:
            hparams["diff_speedup"] = round(hparams["timesteps"] / args.steps)
        hparams["sampling_steps"] = args.steps

    # Process speaker mixture

    spk_mix = parse_commandline_spk_mix(args.spk) if hparams["use_spk_id"] and args.spk is not None else None

    # Update parameters
    for param in params:
        if args.gender is not None and hparams["use_key_shift_embed"]:
            param["gender"] = args.gender
        if spk_mix is not None:
            param["spk_mix"] = spk_mix

    return params, spk_mix


def run_acoustic_inference(args: DiffsingerArguments):
    """Run acoustic model inference"""
    # Process input/output paths
    proj_path = Path(args.proj)
    name = proj_path.stem if not args.title else args.title
    out_path = Path(args.out) if args.out else proj_path.parent
    print(proj_path)
    # Load parameters
    with open(proj_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, list):
        params = [params]
    if len(params) == 0:
        print("The input file is empty.")
        return

    # Process key transition
    if args.key != 0:
        params = trans_key(params, args.key)
        if not args.title:
            name += f"{args.key:+d}key"
        print(f"| key transition: {args.key:+d}")

    # Setup model
    sys.argv = [sys.argv[0], "--exp_name", args.exp, "--infer"]

    set_hparams()

    # Process hyperparameters

    params, spk_mix = process_hparams(hparams, args, params)

    # Run inference
    infer_ins = DiffSingerAcousticInfer(load_vocoder=not args.mel, ckpt_steps=args.ckpt)
    print(f"| Model: {type(infer_ins.model)}")

    try:
        infer_ins.run_inference(
            params, out_dir=out_path, title=name, num_runs=args.num, spk_mix=spk_mix, seed=args.seed, save_mel=args.mel
        )
    except KeyboardInterrupt:
        sys.exit(-1)


def main():
    """Main entry point"""
    parser = PdArgumentParser(DiffsingerArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Validate and process experiment name
    args.exp = find_exp(args.exp)

    # Run inference
    run_acoustic_inference(args)


if __name__ == "__main__":
    main()
