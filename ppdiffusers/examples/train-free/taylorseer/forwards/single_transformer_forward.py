import paddle
from typing import Any, Dict, Optional, Tuple, Union
from ppdiffusers.models.transformer_flux import FluxSingleTransformerBlock
from taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init
def taylorseer_flux_single_block_forward(
    self: FluxSingleTransformerBlock,
    hidden_states: paddle.Tensor,
    temb: paddle.Tensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    gate = gate.unsqueeze(1)

    residual = hidden_states
    
    if current['type'] == 'full':

        current['module'] = 'total'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            #**joint_attention_kwargs,
        )

        hidden_states = paddle.concat([attn_output, mlp_hidden_states], axis=2)

        hidden_states = self.proj_out(hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)


    elif current['type'] == 'Taylor':

        current['module'] = 'total'
        hidden_states = taylor_formula(cache_dic=cache_dic, current=current)

    hidden_states = gate * hidden_states
    hidden_states = residual + hidden_states

    if hidden_states.dtype == paddle.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states
