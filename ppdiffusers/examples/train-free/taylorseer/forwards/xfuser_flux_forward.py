import logging
import time
import paddle
import paddle.distributed as dist
from ppdiffusers.transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)

from typing import Any, Dict, Optional, Tuple, Union
from ppdiffusers import DiffusionPipeline
from ppdiffusers.models import FluxTransformer2DModel
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_paddle_version, logging, scale_lora_layers, unscale_lora_layers
import paddle
import numpy as np
from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

from cache_functions import cache_init, cal_type

logger = logging.get_logger(__name__)

def taylorseer_xfuser_flux_forward(
    self: FluxTransformer2DModel,
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
        hidden_states (`paddle.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`paddle.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`paddle.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `paddle.LongTensor`):
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
        joint_attention_kwargs['cache_dic'], joint_attention_kwargs['current'] = cache_init(self)

    cal_type(joint_attention_kwargs['cache_dic'], joint_attention_kwargs['current'])

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    if is_pipeline_first_stage():
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
    if is_pipeline_first_stage():
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d paddle.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d paddle Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d paddle.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d paddle Tensor"
        )
        img_ids = img_ids[0]

    ids = paddle.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    joint_attention_kwargs['current']['stream'] = 'double_stream'

    for index_block, block in enumerate(self.transformer_blocks):

        joint_attention_kwargs['current']['layer'] = index_block

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward
            
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_paddle_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states = (
                paddle.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            
        # controlnet residual
        # if controlnet_block_samples is not None:
        #     interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
        #     interval_control = int(np.ceil(interval_control))
        #     hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    # if self.stage_info.after_flags["transformer_blocks"]:
    hidden_states = paddle.cat([encoder_hidden_states, hidden_states], dim=1)
    
    joint_attention_kwargs['current']['stream'] = 'single_stream'

    for index_block, block in enumerate(self.single_transformer_blocks):

        joint_attention_kwargs['current']['layer'] = index_block

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_paddle_version(">=", "1.11.0") else {}
            )
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
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        # if controlnet_single_block_samples is not None:
        #     interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
        #     interval_control = int(np.ceil(interval_control))
        #     hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
        #         hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        #         + controlnet_single_block_samples[index_block // interval_control]
        #     )

    encoder_hidden_states = hidden_states[:, : encoder_hidden_states.shape[1], ...]
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    if self.stage_info.after_flags["single_transformer_blocks"]:
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states), None
    else:
        output = hidden_states, encoder_hidden_states

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    joint_attention_kwargs['current']['step'] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)