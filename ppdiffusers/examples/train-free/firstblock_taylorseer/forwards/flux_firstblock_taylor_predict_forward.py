from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import  Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers
from cache_functions import cache_init_step
from taylorseer_utils import step_taylor_formula,step_derivative_approximation
from first_block_utils import all_reduce_sync



def are_two_tensors_similar(t1, t2, *, threshold, parallelized=False):
    if threshold <= 0.0:
        return False

    if t1.shape != t2.shape:
        return False

    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    if parallelized:
        mean_diff = all_reduce_sync(mean_diff, "avg")
        mean_t1 =all_reduce_sync(mean_t1, "avg")
    diff = mean_diff / mean_t1
    return diff.item() < threshold
def FirstBlock_taylor_predict_Forward(
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
            joint_attention_kwargs['cache_dic'], joint_attention_kwargs['current'] = cache_init_step(self)

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

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            encoder_hidden_states, hidden_states= self.transformer_blocks[0](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
               # joint_attention_kwargs=joint_attention_kwargs,
            )
            first_hidden_states_residual = hidden_states - inp 
            if self.downsample_factor > 1:
                first_hidden_states_residual = first_hidden_states_residual[..., ::self.downsample_factor]
            
            can_use_cache = self.prev_first_hidden_states_residual is not None and are_two_tensors_similar(
                self.prev_first_hidden_states_residual,
                first_hidden_states_residual,
                threshold=self.residual_diff_threshold,
                parallelized=False,
            )
            if self.cnt == 0 or self.cnt == self.num_steps - 1:
                should_calc = True
                self.prev_first_hidden_states_residual = None
                
            else:
                should_calc = not can_use_cache
                if can_use_cache == False:
                    self.prev_first_hidden_states_residual = first_hidden_states_residual.clone()
                
                # if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                #     should_calc = False
                # else:
                #     should_calc = True
                #     self.accumulated_rel_l1_distance = 0
            
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0

        cache_dic = joint_attention_kwargs['cache_dic']
        current = joint_attention_kwargs['current']
        if self.enable_teacache:
            if not should_calc:
                #hidden_states += self.previous_residual
                hidden_states = step_taylor_formula(cache_dic=cache_dic, current=current)

            else:
                # ori_hidden_states = hidden_states.clone()
                current['activated_steps'].append(current['step'])
                for index_block, block in enumerate(self.transformer_blocks):
                    #因为上面已经算了现在就不应该算了
                    if index_block == 0:
                        continue
                    if self.training and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = (
                            {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        )
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
                           # joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
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
                hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if self.training and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = (
                            {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
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
                            #joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                step_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
                # self.previous_residual = hidden_states - ori_hidden_states
        # else:
        #     for index_block, block in enumerate(self.transformer_blocks):
        #         if self.training and self.gradient_checkpointing:

        #             def create_custom_forward(module, return_dict=None):
        #                 def custom_forward(*inputs):
        #                     if return_dict is not None:
        #                         return module(*inputs, return_dict=return_dict)
        #                     else:
        #                         return module(*inputs)

        #                 return custom_forward

        #             ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #             encoder_hidden_states, hidden_states = paddle.utils.checkpoint.checkpoint(
        #                 create_custom_forward(block),
        #                 hidden_states,
        #                 encoder_hidden_states,
        #                 temb,
        #                 image_rotary_emb,
        #                 **ckpt_kwargs,
        #             )

        #         else:
        #             encoder_hidden_states, hidden_states = block(
        #                 hidden_states=hidden_states,
        #                 encoder_hidden_states=encoder_hidden_states,
        #                 temb=temb,
        #                 image_rotary_emb=image_rotary_emb,
        #                 joint_attention_kwargs=joint_attention_kwargs,
        #             )

        #         # controlnet residual
        #         if controlnet_block_samples is not None:
        #             interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
        #             interval_control = int(np.ceil(interval_control))
        #             # For Xlabs ControlNet.
        #             if controlnet_blocks_repeat:
        #                 hidden_states = (
        #                     hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
        #                 )
        #             else:
        #                 hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        #     hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        #     for index_block, block in enumerate(self.single_transformer_blocks):
        #         if self.training and self.gradient_checkpointing:

        #             def create_custom_forward(module, return_dict=None):
        #                 def custom_forward(*inputs):
        #                     if return_dict is not None:
        #                         return module(*inputs, return_dict=return_dict)
        #                     else:
        #                         return module(*inputs)

        #                 return custom_forward

        #             ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #             hidden_states = paddle.utils.checkpoint.checkpoint(
        #                 create_custom_forward(block),
        #                 hidden_states,
        #                 temb,
        #                 image_rotary_emb,
        #                 **ckpt_kwargs,
        #             )

        #         else:
        #             hidden_states = block(
        #                 hidden_states=hidden_states,
        #                 temb=temb,
        #                 image_rotary_emb=image_rotary_emb,
        #                 joint_attention_kwargs=joint_attention_kwargs,
        #             )

        #         # controlnet residual
        #         if controlnet_single_block_samples is not None:
        #             interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
        #             interval_control = int(np.ceil(interval_control))
        #             hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
        #                 hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        #                 + controlnet_single_block_samples[index_block // interval_control]
        #             )
        #     hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        joint_attention_kwargs['current']['step'] += 1
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)