# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
import sys
from ast import literal_eval
from typing import Dict, List, Tuple

import paddle

from ..utils import PrecisionType
from ..utils.other import is_port_in_use
from .dataclasses import DistributedType


def _filter_args(args, parser, default_args=[]):
    """
    Filters out all `accelerate` specific args
    """
    new_args, _ = parser.parse_known_args(default_args)
    for key, value in vars(args).items():
        if key in vars(new_args).keys():
            setattr(new_args, key, value)
    return new_args


def prepare_simple_launcher_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct simple launcher environment variables.
    """
    cmd = []
    if args.no_python and args.module:
        raise ValueError("--module and --no_python cannot be used together")
    if not args.no_python:
        cmd.append(sys.executable)
        if args.module:
            cmd.append("-m")
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    current_env["ACCELERATE_USE_CPU"] = str(args.cpu or args.use_cpu)
    if args.debug:
        current_env["ACCELERATE_DEBUG_MODE"] = "true"
    if args.gpu_ids != "all" and args.gpu_ids is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.num_machines > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip
        current_env["MASTER_PORT"] = str(args.main_process_port)
    elif args.num_processes > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip if args.main_process_ip is not None else "127.0.0.1"
        current_env["MASTER_PORT"] = str(args.main_process_port) if args.main_process_port is not None else "29500"

    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)

    return cmd, current_env


def prepare_multi_gpu_env(args: argparse.Namespace) -> Dict[str, str]:
    """
    Prepares and returns an environment with the correct multi-GPU environment variables.
    """
    num_processes = getattr(args, "num_processes")
    num_machines = getattr(args, "num_machines")
    main_process_ip = getattr(args, "main_process_ip")
    main_process_port = getattr(args, "main_process_port")
    if num_machines > 1:
        setattr(args, "nproc_per_node", str(num_processes // num_machines))
        setattr(args, "nnodes", str(num_machines))
        setattr(args, "node_rank", int(args.machine_rank))
        if getattr(args, "same_network", False):
            setattr(args, "master_addr", str(main_process_ip))
            setattr(args, "master_port", str(main_process_port))
        else:
            setattr(args, "rdzv_endpoint", f"{main_process_ip}:{main_process_port}")
    else:
        setattr(args, "nproc_per_node", str(num_processes))
        if main_process_port is not None:
            setattr(args, "master_port", str(main_process_port))

    if main_process_port is None:
        main_process_port = 29500

    # only need to check port availability in main process, in case we have to start multiple launchers on the same machine
    # for some reasons like splitting log files.
    need_port_check = num_machines <= 1 or int(args.machine_rank) == 0
    if need_port_check and is_port_in_use(main_process_port):
        raise ConnectionError(
            f"Tried to launch distributed communication on port `{main_process_port}`, but another process is utilizing it. "
            "Please specify a different port (such as using the `----main_process_port` flag or specifying a different `main_process_port` in your config file)"
            " and rerun your script. To automatically use the next open port (on a single node), you can set this to `0`."
        )

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        setattr(args, "module", True)
    elif args.no_python:
        setattr(args, "no_python", True)

    current_env = os.environ.copy()
    if args.debug:
        current_env["ACCELERATE_DEBUG_MODE"] = "true"
    gpu_ids = getattr(args, "gpu_ids", "all")
    if gpu_ids != "all" and args.gpu_ids is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    mixed_precision = args.mixed_precision.lower()
    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}.")

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)
    return current_env


def _convert_nargs_to_dict(nargs: List[str]) -> Dict[str, str]:
    if len(nargs) < 0:
        return {}
    # helper function to infer type for argsparser

    def _infer_type(s):
        try:
            s = float(s)

            if s // 1 == s:
                return int(s)
            return s
        except ValueError:
            return s

    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args(nargs)
    for index, argument in enumerate(unknown):
        if argument.startswith(("-", "--")):
            action = None
            if index + 1 < len(unknown):  # checks if next index would be in list
                if unknown[index + 1].startswith(("-", "--")):  # checks if next element is an key
                    # raise an error if element is store_true or store_false
                    raise ValueError(
                        "SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types"
                    )
            else:  # raise an error if last element is store_true or store_false
                raise ValueError(
                    "SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types"
                )
            # adds argument to parser based on action_store true
            if action is None:
                parser.add_argument(argument, type=_infer_type)
            else:
                parser.add_argument(argument, action=action)

    return {
        key: (literal_eval(value) if value in ("True", "False") else value)
        for key, value in parser.parse_args(nargs).__dict__.items()
    }


def env_var_path_add(env_var_name, path_to_add):
    """
    Extends a path-based environment variable's value with a new path and returns the updated value. It's up to the
    caller to set it in os.environ.
    """
    paths = [p for p in os.environ.get(env_var_name, "").split(":") if len(p) > 0]
    paths.append(str(path_to_add))
    return ":".join(paths)


class PrepareForLaunch:
    """
    Prepare a function that will launched in a distributed setup.

    Args:
        launcher (`Callable`):
            The function to launch.
        distributed_type ([`~state.DistributedType`]):
            The distributed type to prepare for.
        debug (`bool`, *optional*, defaults to `False`):
            Whether or not this is a debug launch.
    """

    def __init__(self, launcher, distributed_type="NO", debug=False):
        self.launcher = launcher
        self.distributed_type = DistributedType(distributed_type)
        self.debug = debug

    def __call__(self, index, *args):
        if self.debug:
            paddle.distributed.init_parallel_env()

        elif self.distributed_type in (DistributedType.MULTI_GPU,):
            # Prepare the environment for torch.distributed
            os.environ["LOCAL_RANK"] = str(index)
            nproc = int(os.environ.get("NPROC", 1))
            node_rank = int(os.environ.get("NODE_RANK", 0))
            os.environ["RANK"] = str(nproc * node_rank + index)

        os.environ["FORK_LAUNCHED"] = str(1)
        self.launcher(*args)
