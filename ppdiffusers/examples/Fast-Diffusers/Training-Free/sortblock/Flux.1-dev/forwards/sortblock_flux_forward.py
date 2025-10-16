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
from cache_functions import cache_init
from taylorseer_utils import derivative_approximation, taylor_formula

from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)


def SortTaylor_forward(
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
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`paddle.Tensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`paddle.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`paddle.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `paddle.Tensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `paddle.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is None:
        joint_attention_kwargs = {}
    if joint_attention_kwargs.get("cache_dic", None) is None:
        joint_attention_kwargs["cache_dic"], joint_attention_kwargs["current"] = cache_init(self)

    # cal_type(joint_attention_kwargs['cache_dic'], joint_attention_kwargs['current'])

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
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

    self.count += 1

    if timestep == 1000:
        self.current_block_residual = [None] * len(self.transformer_blocks)
        self.current_block_encoder_residual = [None] * len(self.transformer_blocks)
        self.current_single_block_residual = [None] * len(self.single_transformer_blocks)
        self.previous_block_residual = [None] * len(self.transformer_blocks)
        self.previous_single_block_residual = [None] * len(self.single_transformer_blocks)
        self.previous_encoder_block_residual = [None] * len(self.single_transformer_blocks)
        self.count = 0
        self.percentage = 1
        self.result_list = []
        self.result_single_list = []

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    is_within_block_range = self.end <= timestep <= self.start
    if is_within_block_range:
        self.step_Num = self.step_Num2
    if timestep < self.end or timestep > self.start:
        self.step_Num = 1

    joint_attention_kwargs["current"]["stream"] = "double_stream"
    cache_dic = joint_attention_kwargs["cache_dic"]
    current = joint_attention_kwargs["current"]
    if self.count % self.step_Num == 0:
        current["activated_steps"].append(current["step"])
    for index_block, block in enumerate(self.transformer_blocks):
        joint_attention_kwargs["current"]["layer"] = index_block

        should_compute_block = self.count % self.step_Num == 0 or (
            self.result_list != [] and self.result_list[index_block] == 1
        )
        if should_compute_block:
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()
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
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            current["module"] = "hidden_states"
            derivative_approximation(
                cache_dic=cache_dic, current=current, feature=hidden_states.clone() - ori_hidden_states
            )
            current["module"] = "encoder_hidden_states"
            derivative_approximation(
                cache_dic=cache_dic, current=current, feature=encoder_hidden_states.clone() - ori_encoder_hidden_states
            )
            if self.count % self.step_Num == 0:
                self.previous_block_residual[index_block] = hidden_states.clone() - ori_hidden_states
                self.previous_encoder_block_residual[index_block] = (
                    encoder_hidden_states.clone() - ori_encoder_hidden_states
                )

        else:
            if self.count % self.step_Num == 1:
                current["module"] = "hidden_states"
                self.current_block_residual[index_block] = taylor_formula(cache_dic=cache_dic, current=current)
                current["module"] = "encoder_hidden_states"
                self.current_block_encoder_residual[index_block] = taylor_formula(cache_dic=cache_dic, current=current)
            current["module"] = "hidden_states"
            hidden_states += taylor_formula(cache_dic=cache_dic, current=current)
            current["module"] = "encoder_hidden_states"
            encoder_hidden_states += taylor_formula(cache_dic=cache_dic, current=current)

    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

    joint_attention_kwargs["current"]["stream"] = "single_stream"

    for index_block, block in enumerate(self.single_transformer_blocks):
        joint_attention_kwargs["current"]["layer"] = index_block
        should_compute_block = self.count % self.step_Num == 0 or (
            self.result_single_list != [] and self.result_single_list[index_block] == 1
        )
        if should_compute_block:

            ori_hidden_states = hidden_states.clone()
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
                )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
            current["module"] = "hidden_states"
            derivative_approximation(
                cache_dic=cache_dic, current=current, feature=hidden_states.clone() - ori_hidden_states
            )
            if self.count % self.step_Num == 0:
                self.previous_single_block_residual[index_block] = hidden_states.clone() - ori_hidden_states
        else:
            if self.count % self.step_Num == 1:
                current["module"] = "hidden_states"
                self.current_single_block_residual[index_block] = taylor_formula(cache_dic=cache_dic, current=current)
            current["module"] = "hidden_states"
            hidden_states += taylor_formula(cache_dic=cache_dic, current=current)

    if self.count % self.step_Num == 1:
        coefficients = [5.67621e-14, -1.36659e-10, 1.16246e-7, -3.97725e-5, 0.00361, 0.56088]
        rescale_func = np.poly1d(coefficients)
        self.percentage = rescale_func(timestep.item()) * self.beta
        cosine_similarities = []
        cosine_single_similarities = []
        for i in range(len(self.transformer_blocks)):
            cosine_similarity = paddle.nn.functional.cosine_similarity(
                self.previous_block_residual[i][:, :, : self.previous_block_residual[i].shape[-1] // 8].to(
                    paddle.float32
                ),
                self.current_block_residual[i][:, :, : self.current_block_residual[i].shape[-1] // 8].to(
                    paddle.float32
                ),
                axis=-1,
            )
            cosine_similarities.append(cosine_similarity.mean().item())
        sorted_cos = sorted(cosine_similarities)
        threshold = sorted_cos[int(len(self.transformer_blocks) * self.percentage)]
        self.result_list = []
        for j in cosine_similarities:
            if j <= threshold:
                self.result_list.append(1)
            else:
                self.result_list.append(0)
        for i in range(len(self.single_transformer_blocks)):
            cosine_similarity = paddle.nn.functional.cosine_similarity(
                self.previous_single_block_residual[i][
                    :, :, : self.previous_single_block_residual[i].shape[-1] // 8
                ].to(paddle.float32),
                self.current_single_block_residual[i][:, :, : self.current_single_block_residual[i].shape[-1] // 8].to(
                    paddle.float32
                ),
                axis=-1,
            )
            cosine_single_similarities.append(cosine_similarity.mean().item())
        sorted_cos = sorted(cosine_single_similarities)
        threshold = sorted_cos[int(len(self.single_transformer_blocks) * self.percentage)]
        self.result_single_list = []
        for j in cosine_single_similarities:
            if j <= threshold:
                self.result_single_list.append(1)
            else:
                self.result_single_list.append(0)

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    joint_attention_kwargs["current"]["step"] += 1
    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
