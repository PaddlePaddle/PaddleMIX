import paddle

from typing import Any, Dict, Optional, Tuple, Union

from ppdiffusers.models.transformer_flux import FluxTransformerBlock
import paddle

from taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init

def taylorseer_flux_double_block_forward(
    self: FluxTransformerBlock,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    temb: paddle.Tensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}

    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    if current['type'] == 'full':

        current['module'] = 'attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        # encoder_hidden_states -> txt
        # hidden_states -> img
        # (encoder_hidden_states, hidden_states) -> total

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            #**joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
            raise NotImplementedError("Not implemented for TaylorSeer yet.") 

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        current['module'] = 'img_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=ff_output)

        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        derivative_approximation(cache_dic=cache_dic, current=current, feature=context_attn_output)
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        current['module'] = 'txt_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=context_ff_output)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == paddle.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    elif current['type'] == 'Taylor':

        current['module'] = 'attn'
        # Attention.
        # symbolic placeholder
        

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'

        attn_output = taylor_formula(cache_dic=cache_dic, current=current)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
    
        current['module'] = 'img_mlp'

        ff_output = taylor_formula(cache_dic=cache_dic, current=current)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
    
        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'

        context_attn_output = taylor_formula(cache_dic=cache_dic, current=current)

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
    
        current['module'] = 'txt_mlp'

        context_ff_output = taylor_formula(cache_dic=cache_dic, current=current)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        if encoder_hidden_states.dtype == paddle.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states