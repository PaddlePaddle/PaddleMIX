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


from typing import Any, Dict, Optional, Union

import numpy as np
import paddle

from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logger,
    scale_lora_layers,
    unscale_lora_layers,
)

# Use only fallback implementations


def fallback_cache_init_step(model, max_order=3, first_enhance=2):
    """
    Fallback implementation of cache_init_step with configurable parameters
    """
    cache_dic = {"cache": {"hidden": {}}, "max_order": max_order, "first_enhance": first_enhance}
    current = {"step": 0, "activated_steps": []}
    return cache_dic, current


def fallback_step_taylor_formula(cache_dic: Dict, current: Dict) -> paddle.Tensor:
    """
    Fallback implementation of step_taylor_formula with configurable parameters
    """
    if len(current["activated_steps"]) < 1:
        return None

    if len(cache_dic["cache"]["hidden"]) == 0:
        return None

    if 0 not in cache_dic["cache"]["hidden"]:
        return None

    # Check first_enhance threshold
    first_enhance = cache_dic.get("first_enhance", 2)
    if current["step"] < first_enhance:
        return None

    try:
        # If we only have one activated step, just return the cached value
        if len(current["activated_steps"]) < 2:
            return cache_dic["cache"]["hidden"][0]

        x = current["step"] - current["activated_steps"][-1]
        output = cache_dic["cache"]["hidden"][0]

        # Get configured max_order
        max_order = cache_dic.get("max_order", 3)

        # Add higher order terms if available, up to max_order
        for i in range(1, min(max_order, len(cache_dic["cache"]["hidden"]))):
            if i in cache_dic["cache"]["hidden"]:
                term = cache_dic["cache"]["hidden"][i] * (x**i)
                if i > 1:
                    # Factorial approximation for higher orders
                    factorial = 1
                    for j in range(1, i + 1):
                        factorial *= j
                    term = term / factorial
                output = output + term

        return output
    except Exception as e:
        print(f"Error in fallback_step_taylor_formula: {e}")
        # Emergency fallback: return the 0th order term if available
        return cache_dic["cache"]["hidden"].get(0, None)


def fallback_step_derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Fallback implementation of step_derivative_approximation
    """
    try:
        # Always store the current feature as 0th order
        cache_dic["cache"]["hidden"][0] = feature

        # Compute first derivative if we have enough history
        if "previous_feature" in cache_dic:
            cache_dic["cache"]["hidden"][1] = feature - cache_dic["previous_feature"]

        # Store current feature for next iteration
        cache_dic["previous_feature"] = feature.clone()

        # Limit cache size for stability using configured max_order
        max_order = cache_dic.get("max_order", 3)
        keys_to_remove = [k for k in cache_dic["cache"]["hidden"].keys() if k >= max_order]
        for k in keys_to_remove:
            del cache_dic["cache"]["hidden"][k]

    except Exception as e:
        print(f"Error in fallback_step_derivative_approximation: {e}")
        # Emergency fallback: just store the current feature
        cache_dic["cache"]["hidden"][0] = feature


def compute_taylor_coefficients(residual_history: list, max_order: int = 3) -> dict:
    """
    Compute Taylor expansion coefficients based on residual history

    Args:
        residual_history: List of residual history [r_t-2, r_t-1, r_t]
        max_order: Maximum Taylor expansion order

    Returns:
        dict: Taylor coefficients {0: f(t), 1: f'(t), 2: f''(t)/2!, ...}
    """
    if len(residual_history) < 2:
        return {}

    coefficients = {}

    # 0-order: current value
    if len(residual_history) >= 1:
        coefficients[0] = residual_history[-1]

    # 1st-order: first derivative (finite difference approximation)
    if len(residual_history) >= 2:
        coefficients[1] = residual_history[-1] - residual_history[-2]

    # 2nd-order: second derivative (second-order finite difference)
    if len(residual_history) >= 3 and max_order >= 2:
        second_diff = (residual_history[-1] - residual_history[-2]) - (residual_history[-2] - residual_history[-3])
        coefficients[2] = second_diff / 2.0  # divide by 2!

    # Higher orders can be extended similarly, but 2nd-order is sufficient in practice

    return coefficients


def taylor_predict_residual(coefficients: dict, steps_ahead: int = 1) -> paddle.Tensor:
    """
    Predict future residuals using Taylor expansion

    Args:
        coefficients: Taylor coefficients dictionary
        steps_ahead: Number of steps to predict ahead

    Returns:
        Predicted residual
    """
    if not coefficients or 0 not in coefficients:
        return None

    predicted = coefficients[0].clone()  # f(t)

    # Add higher order terms
    for order in range(1, len(coefficients)):
        if order in coefficients:
            term = coefficients[order] * (steps_ahead**order)
            predicted = predicted + term

    # Numerical stability check
    if not paddle.isfinite(predicted).all():
        return coefficients[0]  # fallback to current value

    # Limit the magnitude of prediction change to avoid divergence
    max_change_ratio = 0.2  # maximum change ratio
    if 1 in coefficients:
        change_magnitude = paddle.linalg.norm(coefficients[1] * steps_ahead)
        current_magnitude = paddle.linalg.norm(coefficients[0])
        if current_magnitude > 0 and change_magnitude > max_change_ratio * current_magnitude:
            scale_factor = (max_change_ratio * current_magnitude) / change_magnitude
            predicted = coefficients[0] + coefficients[1] * steps_ahead * scale_factor

    return predicted


def apply_polynomial_rescale(rel_change: float) -> float:
    """
    Apply polynomial rescale function (from TeaCache)
    """
    coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
    rescale_func = np.poly1d(coefficients)
    return float(rescale_func(rel_change))


def TeaBlockCacheTaylorForward(
    self,
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
    TeaBlockCache enhanced with Taylor expansion prediction.

    Combines per-block heuristic caching from TeaBlockCache with
    global Taylor expansion prediction from TeaCache methodology.
    """
    # Initialize joint_attention_kwargs and global Taylor cache
    if joint_attention_kwargs is None:
        joint_attention_kwargs = {}

    # Initialize global Taylor cache (like TeaCache)
    if joint_attention_kwargs.get("cache_dic", None) is None:
        # Get Taylor parameters from taylor_cache_system if available
        max_order = 3  # default
        first_enhance = 2  # default
        if hasattr(self, "taylor_cache_system"):
            max_order = self.taylor_cache_system.get("max_order", 3)
            first_enhance = self.taylor_cache_system.get("first_enhance", 2)

        joint_attention_kwargs["cache_dic"], joint_attention_kwargs["current"] = fallback_cache_init_step(
            self, max_order, first_enhance
        )

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = paddle.concat((txt_ids, img_ids), axis=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # Initialize per-block heuristic states for TeaBlockCache
    if not hasattr(self, "block_heuristic_states"):
        self.block_heuristic_states = {}
        self.single_block_heuristic_states = {}

    # Get global Taylor cache references
    cache_dic = joint_attention_kwargs["cache_dic"]
    current = joint_attention_kwargs["current"]

    # Check if we're in cache-enabled time range
    is_within_time_range = self.step_start <= timestep <= self.step_end

    # Reset states at the beginning of generation
    if timestep == 1000 or self.cnt == 0:
        self.block_heuristic_states = {}
        self.single_block_heuristic_states = {}
        self.cnt = 0

    # Global step counter
    self.cnt += 1

    # Force computation at first and last steps
    force_compute = self.cnt == 1 or self.cnt == self.num_steps

    # ===== TeaBlockCache per-block processing with global Taylor prediction =====

    # Check if we can use global Taylor prediction (similar to TeaCache logic)
    use_global_taylor_prediction = False
    if is_within_time_range and not force_compute:
        # Use the first transformer block for global heuristic (like TeaCache)
        if len(self.transformer_blocks) > 0:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            try:
                norm_result = self.transformer_blocks[0].norm1(inp, emb=temb_)
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result

                # Global TeaCache-style heuristic
                if hasattr(self, "previous_modulated_input") and self.previous_modulated_input is not None:
                    rel_change = (
                        (
                            (modulated_inp - self.previous_modulated_input).abs().mean()
                            / self.previous_modulated_input.abs().mean()
                        )
                        .cpu()
                        .item()
                    )

                    # Apply polynomial rescale (from TeaCache)
                    coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                    rescale_func = np.poly1d(coefficients)

                    if not hasattr(self, "accumulated_rel_l1_distance"):
                        self.accumulated_rel_l1_distance = 0

                    self.accumulated_rel_l1_distance += rescale_func(rel_change)

                    # Check if we can use Taylor prediction
                    if self.accumulated_rel_l1_distance < getattr(self, "rel_l1_thresh", 0.3):
                        use_global_taylor_prediction = True
                    else:
                        self.accumulated_rel_l1_distance = 0

                self.previous_modulated_input = modulated_inp.clone()
            except:
                # If anything fails in global heuristic, force computation
                use_global_taylor_prediction = False

    # Apply global Taylor prediction if possible
    # Check if we have enough steps and are past first_enhance threshold
    first_enhance = cache_dic.get("first_enhance", 2)
    can_use_taylor = (
        use_global_taylor_prediction and len(current["activated_steps"]) >= 1 and current["step"] >= first_enhance
    )

    if can_use_taylor:
        predicted_hidden = fallback_step_taylor_formula(cache_dic=cache_dic, current=current)

        if predicted_hidden is not None and paddle.isfinite(predicted_hidden).all():
            # Use Taylor prediction, skip all computation
            hidden_states = predicted_hidden
            # Skip transformer blocks processing
            skip_transformer_computation = True
        else:
            # Taylor prediction failed, force computation
            skip_transformer_computation = False
    else:
        skip_transformer_computation = False

    # Process transformer blocks
    if not skip_transformer_computation:
        # Store original hidden states for Taylor cache update
        # ori_hidden_states = hidden_states.clone()
        current["activated_steps"].append(current["step"])

        for index_block, block in enumerate(self.transformer_blocks):
            # Initialize per-block heuristic state
            if index_block not in self.block_heuristic_states:
                self.block_heuristic_states[index_block] = {
                    "accumulated_distance": 0,
                    "previous_modulated_input": None,
                    "should_compute": True,
                }

            block_heuristic_state = self.block_heuristic_states[index_block]

            # Determine if this block should be computed (TeaBlockCache heuristic)
            should_compute_block = force_compute

            if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
                # Calculate modulated input for change detection
                inp = hidden_states
                temb_ = temb
                try:
                    norm_result = block.norm1(inp, emb=temb_)

                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result

                    # Apply per-block heuristic using polynomial rescale
                    if block_heuristic_state["previous_modulated_input"] is not None:
                        hidden_dim = modulated_inp.shape[-1]
                        C = max(1, hidden_dim // 8)
                        mod_head = modulated_inp[:, :, :C]
                        prev_head = block_heuristic_state["previous_modulated_input"][:, :, :C]

                        rel_change = ((mod_head - prev_head).abs().mean() / prev_head.abs().mean()).cpu().item()

                        # Apply polynomial rescale
                        coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                        rescale_func = np.poly1d(coefficients)
                        block_heuristic_state["accumulated_distance"] += rescale_func(rel_change)

                        if block_heuristic_state["accumulated_distance"] < self.block_rel_l1_thresh:
                            should_compute_block = False
                        else:
                            block_heuristic_state["accumulated_distance"] = 0
                            should_compute_block = True
                    else:
                        should_compute_block = True

                    block_heuristic_state["previous_modulated_input"] = modulated_inp.clone()
                except:
                    should_compute_block = True
            else:
                should_compute_block = True

            if should_compute_block:
                # Compute the block
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = paddle.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        # joint_attention_kwargs=joint_attention_kwargs,  # 移除包含cache_dic的参数
                    )
            else:
                # Use cached output (simple cache reuse for per-block)
                if hasattr(block_heuristic_state, "cached_hidden") and hasattr(
                    block_heuristic_state, "cached_encoder"
                ):
                    hidden_states = block_heuristic_state["cached_hidden"]
                    encoder_hidden_states = block_heuristic_state["cached_encoder"]
                else:
                    # First time caching, compute and store
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        # joint_attention_kwargs=joint_attention_kwargs,  # 移除包含cache_dic的参数
                    )
                    block_heuristic_state["cached_hidden"] = hidden_states.clone()
                    block_heuristic_state["cached_encoder"] = encoder_hidden_states.clone()

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # Concatenate encoder and image hidden states
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # Process single transformer blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            # Initialize per-single-block heuristic state
            if index_block not in self.single_block_heuristic_states:
                self.single_block_heuristic_states[index_block] = {
                    "accumulated_distance": 0,
                    "previous_modulated_input": None,
                    "should_compute": True,
                }

            single_block_heuristic_state = self.single_block_heuristic_states[index_block]

            # Determine if this single block should be computed
            should_compute_block = force_compute

            if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
                # Calculate modulated input for single blocks
                inp = hidden_states
                temb_ = temb
                try:
                    norm_result = block.norm(inp, emb=temb_)

                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result

                    if single_block_heuristic_state["previous_modulated_input"] is not None:
                        hidden_dim = modulated_inp.shape[-1]
                        C = max(1, hidden_dim // 8)
                        mod_head = modulated_inp[:, :, :C]
                        prev_head = single_block_heuristic_state["previous_modulated_input"][:, :, :C]

                        rel_change = ((mod_head - prev_head).abs().mean() / prev_head.abs().mean()).cpu().item()

                        # Apply polynomial rescale
                        coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                        rescale_func = np.poly1d(coefficients)
                        single_block_heuristic_state["accumulated_distance"] += rescale_func(rel_change)

                        if single_block_heuristic_state["accumulated_distance"] < self.single_block_rel_l1_thresh:
                            should_compute_block = False
                        else:
                            single_block_heuristic_state["accumulated_distance"] = 0
                            should_compute_block = True
                    else:
                        should_compute_block = True

                    single_block_heuristic_state["previous_modulated_input"] = modulated_inp.clone()
                except:
                    should_compute_block = True
            else:
                should_compute_block = True

            if should_compute_block:
                # Compute the single block
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = paddle.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        # joint_attention_kwargs=joint_attention_kwargs,  # 移除包含cache_dic的参数
                    )
            else:
                # Use cached output for single blocks
                if hasattr(single_block_heuristic_state, "cached_hidden"):
                    hidden_states = single_block_heuristic_state["cached_hidden"]
                else:
                    # First time caching, compute and store
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        # joint_attention_kwargs=joint_attention_kwargs,  # 移除包含cache_dic的参数
                    )
                    single_block_heuristic_state["cached_hidden"] = hidden_states.clone()

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        # Extract only the image hidden states
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Update global Taylor cache (like TeaCache)
        # Only update Taylor cache if we have enough activated steps
        if len(current["activated_steps"]) >= 1:
            fallback_step_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)

    # Reset counter if we've reached the end
    if self.cnt == self.num_steps:
        self.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    # Update global step counter (like TeaCache)
    joint_attention_kwargs["current"]["step"] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
