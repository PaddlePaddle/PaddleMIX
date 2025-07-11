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

from dataclasses import dataclass
from typing import Callable, Optional

import paddle

from ..models.transformer_flux import FluxTransformer2DModel
from ..utils import logging
from .hooks import HookRegistry, ModelHook

logger = logging.get_logger(__name__)


@dataclass
class TeaBlockCacheTaylorConfig:
    """
    Configuration for TeaBlockCache + Taylor optimization.

    Args:
        step_start (`int`, defaults to `50`):
            Start timestep for TeaBlockCache caching.
        step_end (`int`, defaults to `950`):
            End timestep for TeaBlockCache caching.
        block_cache_start (`int`, defaults to `1`):
            Start block index for transformer blocks caching.
        single_block_cache_start (`int`, defaults to `1`):
            Start block index for single transformer blocks caching.
        block_rel_l1_thresh (`float`, defaults to `2.0`):
            Relative L1 threshold for transformer blocks.
        single_block_rel_l1_thresh (`float`, defaults to `2.0`):
            Relative L1 threshold for single transformer blocks.
        taylor_max_order (`int`, defaults to `1`):
            Maximum order for Taylor expansion.
        taylor_first_enhance (`int`, defaults to `1`):
            First enhance parameter for Taylor cache.
        rel_l1_thresh (`float`, defaults to `2.0`):
            Relative L1 threshold for Taylor cache system.
        num_inference_steps (`int`, defaults to `50`):
            Total number of inference steps.
        current_timestep_callback (`Callable[[], int]`, defaults to `None`):
            A callback function that returns the current inference timestep.
    """

    step_start: int = 50
    step_end: int = 950
    block_cache_start: int = 1
    single_block_cache_start: int = 1
    block_rel_l1_thresh: float = 2.0
    single_block_rel_l1_thresh: float = 2.0
    taylor_max_order: int = 1
    taylor_first_enhance: int = 1
    rel_l1_thresh: float = 2.0
    num_inference_steps: int = 50
    current_timestep_callback: Optional[Callable[[], int]] = None

    def __repr__(self) -> str:
        return (
            f"TeaBlockCacheTaylorConfig(\n"
            f"  step_start={self.step_start},\n"
            f"  step_end={self.step_end},\n"
            f"  block_cache_start={self.block_cache_start},\n"
            f"  single_block_cache_start={self.single_block_cache_start},\n"
            f"  block_rel_l1_thresh={self.block_rel_l1_thresh},\n"
            f"  single_block_rel_l1_thresh={self.single_block_rel_l1_thresh},\n"
            f"  taylor_max_order={self.taylor_max_order},\n"
            f"  taylor_first_enhance={self.taylor_first_enhance},\n"
            f"  rel_l1_thresh={self.rel_l1_thresh},\n"
            f"  num_inference_steps={self.num_inference_steps},\n"
            f"  current_timestep_callback={self.current_timestep_callback}\n"
            ")"
        )


class TeaBlockCacheTaylorState:
    """
    State for TeaBlockCache + Taylor optimization.
    """

    def __init__(self) -> None:
        self.cnt = 0
        self.block_heuristic_states = {}
        self.single_block_heuristic_states = {}
        self.taylor_cache_system = {
            "max_order": 1,
            "first_enhance": 1,
            "cache": {"hidden": {}},
            "activated_steps": [],
            "step_counter": 0,
        }
        # TeaCache global state
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None

    def reset(self):
        self.cnt = 0
        self.block_heuristic_states = {}
        self.single_block_heuristic_states = {}
        self.taylor_cache_system = {
            "max_order": 1,
            "first_enhance": 1,
            "cache": {"hidden": {}},
            "activated_steps": [],
            "step_counter": 0,
        }
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None

    def __repr__(self):
        return (
            f"TeaBlockCacheTaylorState("
            f"cnt={self.cnt}, "
            f"blocks_cached={len(self.block_heuristic_states)}, "
            f"single_blocks_cached={len(self.single_block_heuristic_states)}, "
            f"taylor_activated_steps={len(self.taylor_cache_system['activated_steps'])}"
            ")"
        )


class TeaBlockCacheTaylorHook(ModelHook):
    """A hook that applies TeaBlockCache + Taylor optimization to FluxTransformer2DModel."""

    _is_stateful = True

    def __init__(self, config: TeaBlockCacheTaylorConfig) -> None:
        super().__init__()
        self.config = config

    def initialize_hook(self, module):
        self.state = TeaBlockCacheTaylorState()

        # Apply configuration to the transformer module
        module.cnt = 0
        module.num_steps = self.config.num_inference_steps
        module.step_start = self.config.step_start
        module.step_end = self.config.step_end
        module.block_cache_start = self.config.block_cache_start
        module.single_block_cache_start = self.config.single_block_cache_start
        module.block_rel_l1_thresh = self.config.block_rel_l1_thresh
        module.single_block_rel_l1_thresh = self.config.single_block_rel_l1_thresh

        # Initialize state dictionaries
        module.block_heuristic_states = {}
        module.single_block_heuristic_states = {}

        # Initialize Taylor cache system
        module.enable_teacache = True
        module.rel_l1_thresh = self.config.rel_l1_thresh
        module.taylor_cache_system = {
            "max_order": self.config.taylor_max_order,
            "first_enhance": self.config.taylor_first_enhance,
            "cache": {"hidden": {}},
            "activated_steps": [],
            "step_counter": 0,
        }

        # Store original forward method and replace it
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward

            # Import and set the TeaBlockCache Taylor forward function
            try:
                import os
                import sys

                # Add the examples directory to the path temporarily
                examples_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "examples",
                    "train-free",
                    "teablockcache",
                    "forwards",
                )
                if examples_path not in sys.path:
                    sys.path.insert(0, examples_path)
                # Replace the forward method
                import types

                from teablockcache_taylor_flux_forward import TeaBlockCacheTaylorForward

                module.forward = types.MethodType(TeaBlockCacheTaylorForward, module)

            except ImportError:
                logger.warning("TeaBlockCache Taylor forward implementation not found, keeping original forward")

        return module

    def reset_state(self, module: FluxTransformer2DModel) -> None:
        self.state.reset()
        # Reset module state
        module.cnt = 0
        module.block_heuristic_states = {}
        module.single_block_heuristic_states = {}
        module.taylor_cache_system = {
            "max_order": self.config.taylor_max_order,
            "first_enhance": self.config.taylor_first_enhance,
            "cache": {"hidden": {}},
            "activated_steps": [],
            "step_counter": 0,
        }
        return module


def apply_teablockcache_taylor(module: paddle.nn.Layer, config: TeaBlockCacheTaylorConfig):
    """
    Apply TeaBlockCache + Taylor optimization to a FluxTransformer2DModel.

    TeaBlockCache enhances the standard TeaCache approach by applying heuristic caching
    at a per-block level, combined with global Taylor expansion prediction for improved
    performance and quality.

    Args:
        module (`paddle.nn.Layer`):
            The FluxTransformer2DModel to apply TeaBlockCache + Taylor optimization to.
        config (`TeaBlockCacheTaylorConfig`):
            The configuration to use for TeaBlockCache + Taylor optimization.

    Example:

    ```python
    >>> import paddle
    >>> from ppdiffusers import FluxPipeline, TeaBlockCacheTaylorConfig, apply_teablockcache_taylor

    >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

    >>> config = TeaBlockCacheTaylorConfig(
    ...     step_start=50,
    ...     step_end=950,
    ...     block_cache_start=1,
    ...     single_block_cache_start=1,
    ...     taylor_max_order=1,
    ...     taylor_first_enhance=1,
    ...     current_timestep_callback=lambda: pipe._current_timestep,
    ... )
    >>> apply_teablockcache_taylor(pipe.transformer, config)
    ```
    """
    if not isinstance(module, FluxTransformer2DModel):
        raise ValueError(
            f"TeaBlockCache + Taylor optimization can only be applied to FluxTransformer2DModel, "
            f"but got {type(module)}."
        )

    if config.current_timestep_callback is None:
        raise ValueError(
            "The `current_timestep_callback` function must be provided in the configuration "
            "to apply TeaBlockCache + Taylor optimization."
        )

    logger.info("Applying TeaBlockCache + Taylor optimization to FluxTransformer2DModel")

    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TeaBlockCacheTaylorHook(config)
    registry.register_hook(hook, "teablockcache_taylor")


def remove_teablockcache_taylor(module: paddle.nn.Layer):
    """
    Remove TeaBlockCache + Taylor optimization from a FluxTransformer2DModel.

    Args:
        module (`paddle.nn.Layer`):
            The FluxTransformer2DModel to remove TeaBlockCache + Taylor optimization from.
    """
    if hasattr(module, "_original_forward"):
        module.forward = module._original_forward
        delattr(module, "_original_forward")

    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.remove_hook("teablockcache_taylor", recurse=True)
