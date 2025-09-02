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
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import paddle

from ..models import FluxTransformer2DModel
from ..models.modeling_outputs import Transformer2DModelOutput
from ..utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from .hooks import HookRegistry, ModelHook

logger = logging.get_logger(__name__)


@dataclass
class SortBlockConfig:
    """Configuration for SortBlock optimization.

    Args:
        num_inference_steps (`int`, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a
            higher quality image at the expense of slower inference.
        timestep_start (`int`, defaults to `900`):
            The timestep to start applying SortBlock optimization.
        timestep_end (`int`, defaults to `100`):
            The timestep to end applying SortBlock optimization.
        percentage (`float`, defaults to `1.0`):
            The percentage of blocks to compute in each layer.
        step_num (`int`, defaults to `1`):
            The step number for normal operation.
        step_num2 (`int`, defaults to `5`):
            The step number when within the timestep range.
        beta (`float`, defaults to `0.3`):
            The beta parameter for rescaling.
        current_timestep_callback (`Callable[[], int]`, *optional*):
            A callback function that returns the current inference timestep. This
            is required for timestep-based optimization.
    """

    num_inference_steps: int = 50
    timestep_start: int = 900
    timestep_end: int = 100
    percentage: float = 1.0
    step_num: int = 1
    step_num2: int = 5
    beta: float = 0.3
    current_timestep_callback: Callable[[], int] = None

    def __repr__(self) -> str:
        return (
            "SortBlockConfig(\n"
            f"  num_inference_steps={self.num_inference_steps},\n"
            f"  timestep_start={self.timestep_start},\n"
            f"  timestep_end={self.timestep_end},\n"
            f"  percentage={self.percentage},\n"
            f"  step_num={self.step_num},\n"
            f"  step_num2={self.step_num2},\n"
            f"  beta={self.beta},\n"
            f"  current_timestep_callback={self.current_timestep_callback}\n"
            ")"
        )


class SortBlockState:
    """State for SortBlock optimization.

    Attributes:
        count (`int`):
            The current count for step tracking.
        current_block_residual (`list`):
            Current block residuals for transformer blocks.
        current_block_encoder_residual (`list`):
            Current encoder block residuals for transformer blocks.
        current_single_block_residual (`list`):
            Current single block residuals for single transformer blocks.
        previous_block_residual (`list`):
            Previous block residuals for transformer blocks.
        previous_single_block_residual (`list`):
            Previous single block residuals for single transformer blocks.
        previous_encoder_block_residual (`list`):
            Previous encoder block residuals.
        result_list (`list`):
            List of results for transformer blocks.
        result_single_list (`list`):
            List of results for single transformer blocks.
        percentage (`float`):
            Current percentage value.
    """

    def __init__(self, transformer_blocks_len: int, single_transformer_blocks_len: int):
        self.count = 0
        self.current_block_residual = [None] * transformer_blocks_len
        self.current_block_encoder_residual = [None] * transformer_blocks_len
        self.current_single_block_residual = [None] * single_transformer_blocks_len
        self.previous_block_residual = [None] * transformer_blocks_len
        self.previous_single_block_residual = [None] * single_transformer_blocks_len
        self.previous_encoder_block_residual = [None] * transformer_blocks_len
        self.result_list = []
        self.result_single_list = []
        self.percentage = 1.0

    def reset(self, transformer_blocks_len: int, single_transformer_blocks_len: int):
        self.count = 0
        self.current_block_residual = [None] * transformer_blocks_len
        self.current_block_encoder_residual = [None] * transformer_blocks_len
        self.current_single_block_residual = [None] * single_transformer_blocks_len
        self.previous_block_residual = [None] * transformer_blocks_len
        self.previous_single_block_residual = [None] * single_transformer_blocks_len
        self.previous_encoder_block_residual = [None] * transformer_blocks_len
        self.result_list = []
        self.result_single_list = []
        self.percentage = 1.0

    def __repr__(self):
        return f"SortBlockState(count={self.count}, percentage={self.percentage})"


class SortBlockHook(ModelHook):
    """A hook that applies SortBlock optimization to FluxTransformer2DModel."""

    _is_stateful = True

    def __init__(self, config: SortBlockConfig):
        super().__init__()
        self.config = config

    def initialize_hook(self, module):
        if not isinstance(module, FluxTransformer2DModel):
            raise ValueError("SortBlock optimization can only be applied to FluxTransformer2DModel")

        transformer_blocks_len = len(module.transformer_blocks)
        single_transformer_blocks_len = len(module.single_transformer_blocks)
        self.state = SortBlockState(transformer_blocks_len, single_transformer_blocks_len)

        # Store original forward method
        self.original_forward = module.forward

        # Set configuration attributes on module
        module.num_steps = self.config.num_inference_steps
        module.start = self.config.timestep_start
        module.end = self.config.timestep_end
        module.percentage = self.config.percentage
        module.step_Num = self.config.step_num
        module.step_Num2 = self.config.step_num2
        module.beta = self.config.beta
        module.count = 0

        # Initialize state attributes on module
        module.current_block_residual = self.state.current_block_residual
        module.current_block_encoder_residual = self.state.current_block_encoder_residual
        module.current_single_block_residual = self.state.current_single_block_residual
        module.previous_block_residual = self.state.previous_block_residual
        module.previous_single_block_residual = self.state.previous_single_block_residual
        module.previous_encoder_block_residual = self.state.previous_encoder_block_residual
        module.result_list = self.state.result_list
        module.result_single_list = self.state.result_single_list

        # Store reference to hook for state management
        module._sort_block_hook = self

        # Replace forward method with SortBlock implementation
        def sort_block_forward_wrapper(
            hidden_states: paddle.Tensor,
            encoder_hidden_states: paddle.Tensor = None,
            pooled_projections: paddle.Tensor = None,
            timestep: paddle.Tensor = None,
            img_ids: paddle.Tensor = None,
            txt_ids: paddle.Tensor = None,
            guidance: paddle.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
        ):
            return self._sort_block_forward(
                module,
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                joint_attention_kwargs,
                controlnet_block_samples,
                controlnet_single_block_samples,
                return_dict,
                controlnet_blocks_repeat,
            )

        module.forward = sort_block_forward_wrapper

        return module

    def reset_state(self, module):
        if hasattr(self, "state"):
            transformer_blocks_len = len(module.transformer_blocks)
            single_transformer_blocks_len = len(module.single_transformer_blocks)
            self.state.reset(transformer_blocks_len, single_transformer_blocks_len)

            # Reset module attributes
            module.count = 0
            module.current_block_residual = self.state.current_block_residual
            module.current_block_encoder_residual = self.state.current_block_encoder_residual
            module.current_single_block_residual = self.state.current_single_block_residual
            module.previous_block_residual = self.state.previous_block_residual
            module.previous_single_block_residual = self.state.previous_single_block_residual
            module.previous_encoder_block_residual = self.state.previous_encoder_block_residual
            module.result_list = self.state.result_list
            module.result_single_list = self.state.result_single_list

        return module

    def _derivative_approximation(self, feature, step_diff):
        """Simple derivative approximation for demonstration."""
        if step_diff == 0:
            return feature
        return feature / step_diff

    def _taylor_formula(self, derivative, step_diff):
        """Simple Taylor expansion for demonstration."""
        return derivative * step_diff

    def _sort_block_forward(
        self,
        module,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        pooled_projections: paddle.Tensor = None,
        timestep: paddle.Tensor = None,
        img_ids: paddle.Tensor = None,
        txt_ids: paddle.Tensor = None,
        guidance: paddle.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[paddle.Tensor, Transformer2DModelOutput]:
        """
        SortBlock optimized forward method for FluxTransformer2DModel.
        """
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(module, lora_scale)

        hidden_states = module.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            module.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else module.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = module.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = paddle.concat((txt_ids, img_ids), axis=0)
        image_rotary_emb = module.pos_embed(ids)

        module.count += 1

        # Reset at the beginning of inference
        if timestep == 1000:
            transformer_blocks_len = len(module.transformer_blocks)
            single_transformer_blocks_len = len(module.single_transformer_blocks)
            module.current_block_residual = [None] * transformer_blocks_len
            module.current_block_encoder_residual = [None] * transformer_blocks_len
            module.current_single_block_residual = [None] * single_transformer_blocks_len
            module.previous_block_residual = [None] * transformer_blocks_len
            module.previous_single_block_residual = [None] * single_transformer_blocks_len
            module.previous_encoder_block_residual = [None] * transformer_blocks_len
            module.count = 0
            module.percentage = 1.0
            module.result_list = []
            module.result_single_list = []

        # Determine step number based on timestep range
        is_within_block_range = module.end <= timestep <= module.start
        if is_within_block_range:
            module.step_Num = module.step_Num2
        else:
            module.step_Num = 1

        # Process transformer blocks
        for index_block, block in enumerate(module.transformer_blocks):
            should_compute_block = module.count % module.step_Num == 0 or (
                module.result_list != [] and module.result_list[index_block] == 1
            )

            if should_compute_block:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()

                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

                # Apply controlnet residuals if needed
                if controlnet_block_samples is not None:
                    interval_control = len(module.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

                # Store residuals for future approximations
                if module.count % module.step_Num == 0:
                    module.previous_block_residual[index_block] = hidden_states.clone() - ori_hidden_states
                    module.previous_encoder_block_residual[index_block] = (
                        encoder_hidden_states.clone() - ori_encoder_hidden_states
                    )
            else:
                # Use Taylor approximation
                if module.count % module.step_Num == 1 and index_block < len(module.previous_block_residual):
                    if module.previous_block_residual[index_block] is not None:
                        module.current_block_residual[index_block] = module.previous_block_residual[index_block]
                    if module.previous_encoder_block_residual[index_block] is not None:
                        module.current_block_encoder_residual[index_block] = module.previous_encoder_block_residual[
                            index_block
                        ]

                # Apply approximated residuals
                if (
                    index_block < len(module.previous_block_residual)
                    and module.previous_block_residual[index_block] is not None
                ):
                    hidden_states += module.previous_block_residual[index_block]
                if (
                    index_block < len(module.previous_encoder_block_residual)
                    and module.previous_encoder_block_residual[index_block] is not None
                ):
                    encoder_hidden_states += module.previous_encoder_block_residual[index_block]

        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # Process single transformer blocks
        for index_block, block in enumerate(module.single_transformer_blocks):
            should_compute_block = module.count % module.step_Num == 0 or (
                module.result_single_list != [] and module.result_single_list[index_block] == 1
            )

            if should_compute_block:
                ori_hidden_states = hidden_states.clone()

                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

                # Apply controlnet residuals if needed
                if controlnet_single_block_samples is not None:
                    interval_control = len(module.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

                if module.count % module.step_Num == 0:
                    module.previous_single_block_residual[index_block] = hidden_states.clone() - ori_hidden_states
            else:
                # Use Taylor approximation
                if module.count % module.step_Num == 1 and index_block < len(module.previous_single_block_residual):
                    if module.previous_single_block_residual[index_block] is not None:
                        module.current_single_block_residual[index_block] = module.previous_single_block_residual[
                            index_block
                        ]

                # Apply approximated residuals
                if (
                    index_block < len(module.previous_single_block_residual)
                    and module.previous_single_block_residual[index_block] is not None
                ):
                    hidden_states += module.previous_single_block_residual[index_block]

        # Compute similarity and update result lists
        if module.count % module.step_Num == 1:
            coefficients = [5.67621e-14, -1.36659e-10, 1.16246e-7, -3.97725e-5, 0.00361, 0.56088]
            rescale_func = np.poly1d(coefficients)
            module.percentage = rescale_func(timestep.item()) * module.beta

            # Compute cosine similarities for transformer blocks
            cosine_similarities = []
            for i in range(len(module.transformer_blocks)):
                if module.previous_block_residual[i] is not None and module.current_block_residual[i] is not None:
                    cosine_similarity = paddle.nn.functional.cosine_similarity(
                        module.previous_block_residual[i][:, :, : module.previous_block_residual[i].shape[-1] // 8].to(
                            paddle.float32
                        ),
                        module.current_block_residual[i][:, :, : module.current_block_residual[i].shape[-1] // 8].to(
                            paddle.float32
                        ),
                        axis=-1,
                    )
                    cosine_similarities.append(cosine_similarity.mean().item())
                else:
                    cosine_similarities.append(1.0)  # Default to high similarity

            if cosine_similarities:
                sorted_cos = sorted(cosine_similarities)
                threshold = sorted_cos[int(len(module.transformer_blocks) * module.percentage)]
                module.result_list = [1 if j <= threshold else 0 for j in cosine_similarities]

            # Compute cosine similarities for single transformer blocks
            cosine_single_similarities = []
            for i in range(len(module.single_transformer_blocks)):
                if (
                    module.previous_single_block_residual[i] is not None
                    and module.current_single_block_residual[i] is not None
                ):
                    cosine_similarity = paddle.nn.functional.cosine_similarity(
                        module.previous_single_block_residual[i][
                            :, :, : module.previous_single_block_residual[i].shape[-1] // 8
                        ].to(paddle.float32),
                        module.current_single_block_residual[i][
                            :, :, : module.current_single_block_residual[i].shape[-1] // 8
                        ].to(paddle.float32),
                        axis=-1,
                    )
                    cosine_single_similarities.append(cosine_similarity.mean().item())
                else:
                    cosine_single_similarities.append(1.0)  # Default to high similarity

            if cosine_single_similarities:
                sorted_cos = sorted(cosine_single_similarities)
                threshold = sorted_cos[int(len(module.single_transformer_blocks) * module.percentage)]
                module.result_single_list = [1 if j <= threshold else 0 for j in cosine_single_similarities]

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = module.norm_out(hidden_states, temb)
        output = module.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(module, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def apply_sort_block(module: paddle.nn.Layer, config: SortBlockConfig):
    """
    Apply SortBlock optimization to a given FluxTransformer2DModel.

    SortBlock is an optimization method that uses Taylor series approximation to skip certain transformer
    block computations during inference, reducing computational cost while maintaining output quality.

    Args:
        module (`paddle.nn.Layer`):
            The FluxTransformer2DModel module to apply SortBlock optimization to.
        config (`SortBlockConfig`):
            The configuration to use for SortBlock optimization.

    Example:

    ```python
    >>> import paddle
    >>> from ppdiffusers import FluxPipeline, SortBlockConfig, apply_sort_block

    >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)

    >>> config = SortBlockConfig(
    ...     num_inference_steps=50,
    ...     timestep_start=900,
    ...     timestep_end=100,
    ...     beta=0.3,
    ...     current_timestep_callback=lambda: pipe._current_timestep,
    ... )
    >>> apply_sort_block(pipe.transformer, config)
    ```
    """
    if not isinstance(module, FluxTransformer2DModel):
        raise ValueError("SortBlock optimization can only be applied to FluxTransformer2DModel")
        if config.current_timestep_callback is None:
            logger.warning(
                "The `current_timestep_callback` function is not provided. "
                "SortBlock may not work optimally without access to the current "
                "timestep information."
            )

    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = SortBlockHook(config)
    registry.register_hook(hook, "sort_block")

    logger.info("SortBlock optimization has been applied to the FluxTransformer2DModel")
