from typing import Any, Dict, Optional, List, Union
from ppdiffusers import DiffusionPipeline
from ppdiffusers.models import SD3Transformer2DModel
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers, recompute_use_reentrant, use_old_recompute
from paddle.distributed.fleet.utils import recompute
import paddle
import numpy as np


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def teacache_forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        pooled_projections: paddle.Tensor = None,
        timestep: paddle.Tensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[paddle.Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.
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

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            logger.debug(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if self.inference_optimize:
            hidden_states = self.simplified_sd3(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )
            encoder_hidden_states = None
        else:
            if self.enable_teacache:
                inp = hidden_states.clone()
                temb_ = temb.clone()
                modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
                if self.cnt == 0 or self.cnt == self.num_steps-1:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                else: 
                    coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
                    rescale_func = np.poly1d(coefficients)
                    self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance = 0
                self.previous_modulated_input = modulated_inp 
                self.cnt += 1 
                if self.cnt == self.num_steps:
                    self.cnt = 0  
            
            if self.enable_teacache:
                if not should_calc:
                    hidden_states += self.previous_residual
                else:
                    ori_hidden_states = hidden_states.clone()

                    for index_block, block in enumerate(self.transformer_blocks):
                        if self.training and self.gradient_checkpointing and not use_old_recompute():

                            def create_custom_forward(module, return_dict=None):
                                def custom_forward(*inputs):
                                    if return_dict is not None:
                                        return module(*inputs, return_dict=return_dict)
                                    else:
                                        return module(*inputs)

                                return custom_forward

                            ckpt_kwargs = {} if recompute_use_reentrant() else {"use_reentrant": False}
                            hidden_states = recompute(
                                create_custom_forward(block),
                                hidden_states,
                                encoder_hidden_states,
                                temb,
                                **ckpt_kwargs,
                            )
                        else:
                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                            )
                        
                        # controlnet residual
                        if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                            interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                            hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

                    self.previous_residual = hidden_states - ori_hidden_states

            else:
                for index_block, block in enumerate(self.transformer_blocks):
                    if self.training and self.gradient_checkpointing and not use_old_recompute():

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs = {} if recompute_use_reentrant() else {"use_reentrant": False}
                        hidden_states = recompute(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            **ckpt_kwargs,
                        )
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                        )
                    
                    # controlnet residual
                    if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                        interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                        hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )

        hidden_states = paddle.transpose(hidden_states, [0, 5, 1, 3, 2, 4])
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

SD3Transformer2DModel.forward = teacache_forward
num_inference_steps = 28
seed = 42
prompt = "An image of a squirrel in Picasso style"
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", paddle_dtype=paddle.float16)
# TeaCache
pipeline.transformer.__class__.enable_teacache = True
pipeline.transformer.__class__.cnt = 0
pipeline.transformer.__class__.num_steps = num_inference_steps
pipeline.transformer.__class__.rel_l1_thresh = 0.6 # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
pipeline.transformer.__class__.previous_modulated_input = None
pipeline.transformer.__class__.previous_residual = None


img = pipeline(
    prompt, 
    num_inference_steps=num_inference_steps,
    generator=paddle.Generator().manual_seed(seed),
    ).images[0]
img.save("{}.png".format('TeaCache_' + prompt))