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

import random
from pathlib import Path
from typing import List

import numpy as np
import paddle
from paddle.amp import GradScaler

from .utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SAFE_MODEL_NAME,
    SAFE_WEIGHTS_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    WEIGHTS_NAME,
    get_pretty_name,
    save,
)

# isort: off
from .logging import get_logger
from .state import PartialState

# isort: on

logger = get_logger(__name__)


def save_accelerator_state(
    output_dir: str,
    model_states: List[dict],
    optimizers: list,
    schedulers: list,
    dataloaders: list,
    process_index: int,
    scaler: GradScaler = None,
    save_on_each_node: bool = False,
    safe_serialization: bool = True,
):
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.

    <Tip>

    If `safe_serialization` is `True`, models will be saved with `safetensors` while the rest are saved using native
    `pickle`.

    </Tip>

    Args:
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        model_states (`List[paddle.nn.Layer]`):
            A list of model states
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        dataloaders (`List[torch.utils.data.DataLoader]`):
            A list of dataloader instances to save their sampler states
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional gradient scaler instance to save
        save_on_each_node (`bool`, *optional*):
            Whether to save on every node, or only the main node.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    output_dir = Path(output_dir)
    # Model states
    for i, state in enumerate(model_states):
        weights_name = WEIGHTS_NAME if not safe_serialization else SAFE_WEIGHTS_NAME
        if i > 0:
            weights_name = weights_name.replace(".", f"_{i}.")
        output_model_file = str(output_dir.joinpath(weights_name))
        save(state, output_model_file, save_on_each_node=save_on_each_node, safe_serialization=safe_serialization)
        logger.info(f"Model weights saved in {output_model_file}")
    # Optimizer states
    for i, opt in enumerate(optimizers):
        state = opt.state_dict()
        optimizer_name = f"{OPTIMIZER_NAME}.pdparams" if i == 0 else f"{OPTIMIZER_NAME}_{i}.pdparams"
        output_optimizer_file = str(output_dir.joinpath(optimizer_name))
        save(state, output_optimizer_file, save_on_each_node=save_on_each_node, safe_serialization=False)
        logger.info(f"Optimizer state saved in {output_optimizer_file}")
    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        state = scheduler.state_dict()
        scheduler_name = f"{SCHEDULER_NAME}.pdparams" if i == 0 else f"{SCHEDULER_NAME}_{i}.pdparams"
        output_scheduler_file = str(output_dir.joinpath(scheduler_name))
        save(state, output_scheduler_file, save_on_each_node=save_on_each_node, safe_serialization=False)
        logger.info(f"Scheduler state saved in {output_scheduler_file}")

    # GradScaler state
    if scaler is not None:
        state = scaler.state_dict()
        output_scaler_file = str(output_dir.joinpath(SCALER_NAME))
        paddle.save(state, output_scaler_file)
        logger.info(f"Gradient scaler state saved in {output_scaler_file}")
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["paddle_manual_seed"] = paddle.get_rng_state()
    output_states_file = str(output_dir.joinpath(states_name))
    paddle.save(states, output_states_file)
    logger.info(f"Random states saved in {output_states_file}")
    return output_dir


def load_accelerator_state(
    input_dir,
    models,
    optimizers,
    schedulers,
    dataloaders,
    process_index,
    scaler=None,
    map_location=None,
    **load_model_func_kwargs,
):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        input_dir (`str` or `os.PathLike`):
            The name of the folder to load all relevant weights and states.
        models (`List[paddle.nn.Layer]`):
            A list of model instances
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional *GradScaler* instance to load
        map_location (`str`, *optional*):
            What device to load the optimizer state onto. Should be one of either "cpu" or "on_device".
        load_model_func_kwargs (`dict`, *optional*):
            Additional arguments that can be passed to the model's `set_state_dict` method.
    """
    if map_location not in [None, "cpu", "on_device"]:
        raise TypeError(
            "Unsupported optimizer map location passed, please choose one of `None`, `'cpu'`, or `'on_device'`"
        )
    if map_location is None:
        map_location = "on_device"
    elif map_location == "on_device":  # default on device
        map_location = PartialState().device

    input_dir = Path(input_dir)
    # Model states
    for i, model in enumerate(models):
        ending = f"_{i}" if i > 0 else ""
        input_model_file = input_dir.joinpath(f"{SAFE_MODEL_NAME}{ending}.safetensors")
        if input_model_file.exists():
            from safetensors.paddle import load_file

            state_dict = load_file(str(input_model_file), device=str(map_location))
        else:
            # Load with torch
            input_model_file = input_dir.joinpath(f"{MODEL_NAME}{ending}.pdparams")
            state_dict = paddle.load(str(input_model_file), return_numpy=map_location == "cpu")
        models[i].set_state_dict(state_dict, **load_model_func_kwargs)
    logger.info("All model weights loaded successfully")

    # Optimizer states
    for i, opt in enumerate(optimizers):
        optimizer_name = f"{OPTIMIZER_NAME}.pdparams" if i == 0 else f"{OPTIMIZER_NAME}_{i}.pdparams"
        input_optimizer_file = input_dir.joinpath(optimizer_name)
        optimizer_state = paddle.load(str(input_optimizer_file), return_numpy=map_location == "cpu")
        optimizers[i].set_state_dict(optimizer_state)
    logger.info("All optimizer states loaded successfully")

    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        scheduler_name = f"{SCHEDULER_NAME}.pdparams" if i == 0 else f"{SCHEDULER_NAME}_{i}.pdparams"
        input_scheduler_file = input_dir.joinpath(scheduler_name)
        scheduler.set_state_dict(paddle.load(str(input_scheduler_file)))
    logger.info("All scheduler states loaded successfully")

    # GradScaler state
    if scaler is not None:
        input_scaler_file = input_dir.joinpath(SCALER_NAME)
        scaler.set_state_dict(paddle.load(str(input_scaler_file)))
        logger.info("GradScaler state loaded successfully")

    # Random states
    try:
        states = paddle.load(str(input_dir.joinpath(f"{RNG_STATE_NAME}_{process_index}.pkl")))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        paddle.set_cuda_rng_state(states["paddle_manual_seed"])
        logger.info("All random states loaded successfully")
    except Exception:
        logger.info("Could not load random states")


def save_custom_state(obj, path, index: int = 0, save_on_each_node: bool = False):
    """
    Saves the state of `obj` to `{path}/custom_checkpoint_{index}.pkl`
    """
    # Should this be the right way to get a qual_name type value from `obj`?
    save_location = Path(path) / f"custom_checkpoint_{index}.pkl"
    logger.info(f"Saving the state of {get_pretty_name(obj)} to {save_location}")
    save(obj.state_dict(), save_location, save_on_each_node=save_on_each_node)


def load_custom_state(obj, path, index: int = 0):
    """
    Loads the state of `obj` at `{path}/custom_checkpoint_{index}.pkl`
    """
    load_location = f"{path}/custom_checkpoint_{index}.pkl"
    logger.info(f"Loading the state of {get_pretty_name(obj)} from {load_location}")
    obj.set_state_dict(paddle.load(load_location, return_numpy=True))
