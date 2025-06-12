import paddle
import paddle.nn.functional as F
from ppdiffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    logging
)
import inspect
from typing import Callable, List, Optional, Tuple, Union
logger = logging.get_logger(__name__) 

def register_forward(
    model, 
    filter_name: str = 'Attention', 
    keep_shape: bool = True,
    sa_kward: dict = None,
    ca_kward: dict = None,
    **kwargs,
):
    """
    A customized forward function for cross attention layer. 
    Detailed information in https://github.com/HaozheLiu-ST/T-GATE

    Args:
        model (`paddle.nn.Layer`):
            A diffusion model contains cross attention layers.
        filter_name (`str`):
            The name to filter the selected layer.
        keep_shape (`bool`):
            Whether or not to remain the shape of hidden features
        sa_kward: (`dict`):
            A kwargs dictionary to pass along to the self attention for caching and reusing. 
        ca_kward: (`dict`):
            A kwargs dictionary to pass along to the cross attention for caching and reusing. 

    Returns:
        count (`int`): The number of the cross attention layers used in the given model.
    """

    count = 0
    def warp_custom(
        self: paddle.nn.Layer,
        keep_shape:bool = True,
        ca_kward: dict = None,
        sa_kward: dict = None,
        **kwargs,
    ):
        def forward(
            hidden_states: paddle.Tensor,
            encoder_hidden_states: Optional[paddle.Tensor] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            keep_shape = keep_shape,
            ca_cache = ca_kward['cache'],
            sa_cache = sa_kward['cache'],
            ca_reuse = ca_kward['reuse'],
            sa_reuse = sa_kward['reuse'],
            **cross_attention_kwargs,
        ) -> paddle.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`paddle.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`paddle.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`paddle.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `paddle.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            if not hasattr(self,'cache'):
                self.cache = None
            if not hasattr(self,'cache1'):
                self.cache1 = None
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            
            if len(unused_kwargs) > 0:
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

            states =  tgate_processor_flux(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                keep_shape = keep_shape,
                cache = self.cache,
                cache1 = self.cache1,
                ca_cache = ca_cache,
                sa_cache = sa_cache,
                ca_reuse = ca_reuse,
                sa_reuse = sa_reuse,
                **cross_attention_kwargs,
            )
            if len(states) == 4:
                hidden_states,encoder_hidden_states, cache,cache1 = states
                if cache1 is not None:
                    self.cache1 = cache1
                if cache is not None:
                    self.cache = cache
                return hidden_states, encoder_hidden_states
            elif len(states) == 2:
                hidden_states, cache = states
            if cache is not None:
                self.cache = cache
            return hidden_states
        return forward

    def register_recr(
        net: paddle.nn.Layer, 
        count: int = None, 
        keep_shape:bool = True, 
        ca_kward:dict = None,
        sa_kward:dict = None
    ):
        if net.__class__.__name__ == filter_name:
            net.forward = warp_custom(net, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
            return count + 1
        elif hasattr(net, 'children'):
            for net_child in net.children():
                count = register_recr(net_child, count, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
        return count

    return register_recr(model, count, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward) 


def tgate_processor(
    attn = None,
    hidden_states = None,
    encoder_hidden_states = None,
    attention_mask  = None,
    temb  = None,
    cache = None,
    keep_shape = True,
    ca_cache = False,
    sa_cache = False,
    ca_reuse = False,
    sa_reuse = False,
    *args,
    **kwargs,
) -> paddle.Tensor:

    r"""
    A customized forward function of the `AttnProcessor2_0` class.

    Args:
        hidden_states (`paddle.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`paddle.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`paddle.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `paddle.Tensor`: The output of the attention layer.
    """

    if not hasattr(F, "scaled_dot_product_attention"):
        raise ImportError("Paddle version does not support scaled dot product attention")

    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states

    cross_attn = encoder_hidden_states is not None
    self_attn =  encoder_hidden_states is None

    input_ndim = hidden_states.ndim

    # if the attention is cross-attention or self-attention
    if cross_attn and ca_reuse and cache is not None:
        hidden_states = cache
    elif self_attn and sa_reuse and cache is not None:
        hidden_states = cache
    else:

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.reshape([batch_size, attn.heads, -1, attention_mask.shape[-1]])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim])

        key = key.reshape([batch_size, -1, attn.heads, head_dim])
        value = value.reshape([batch_size, -1, attn.heads, head_dim])

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = hidden_states.cast(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if (cross_attn and ca_cache) or (self_attn and sa_cache):
            if keep_shape:
                cache = hidden_states
            else:
                hidden_uncond, hidden_pred_text = hidden_states.chunk(2)
                cache = (hidden_uncond + hidden_pred_text ) / 2
        else:
            cache = None

    if input_ndim == 4:
        hidden_states = hidden_states.transpose([0, 1, 3, 2]).reshape([batch_size, channel, height, width])

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states, cache

def tgate_processor_flux(
    attn: None,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor = None,
    attention_mask: Optional[paddle.Tensor] = None,
    image_rotary_emb: Optional[paddle.Tensor] = None,
    cache = None,
    cache1 = None,
    keep_shape = True,
    ca_cache = False,
    sa_cache = False,
    ca_reuse = False,
    sa_reuse = False,
    *args,
    **kwargs,
) -> paddle.Tensor:

    r"""
    A customized forward function of the `AttnProcessor2_0` class.
    Args:
        hidden_states (`paddle.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`paddle.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`paddle.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.
    Returns:
        `paddle.Tensor`: The output of the attention layer.
    """

    if not hasattr(F, "scaled_dot_product_attention"):
        raise ImportError("Paddle version does not support scaled dot product attention")

    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states

    self_attn= True
    cross_attn= False

    input_ndim = hidden_states.ndim

   
    if cross_attn and ca_reuse and cache is not None:
        hidden_states = cache
        if cache1 is not None:
            encoder_hidden_states = cache1
    elif self_attn and sa_reuse and cache is not None:
        hidden_states = cache
        if encoder_hidden_states is not None and cache1 is not None:
            encoder_hidden_states = cache1
    else:

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
        key = key.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
        value = value.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = paddle.concat([encoder_hidden_states_query_proj, query], axis=2)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=2)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=2)

        if image_rotary_emb is not None:
            # from models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        hidden_states = F.scaled_dot_product_attention_(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = hidden_states.astype(query.dtype)

        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            
            if (self_attn and sa_cache):
                if keep_shape:
                    cache = hidden_states
                    cache1 = encoder_hidden_states
                else:
                    cache = hidden_states
                    cache1 = encoder_hidden_states
            else:
                cache = None
                cache1 = None

            return hidden_states, encoder_hidden_states,cache,cache1
        else:
            # cache1 = encoder_hidden_states
            if (self_attn and sa_cache):
                if keep_shape:
                    cache = hidden_states
                else:
                    cache = hidden_states
            else:
                cache = None
            return hidden_states,cache
    if encoder_hidden_states is not None:
        return hidden_states, encoder_hidden_states,cache,cache1
    else:
        return hidden_states,cache



def tgate_scheduler(
    cur_step: int = None, 
    gate_step: int = 10, 
    sp_interval: int = 5,
    fi_interval: int = 1,
    warm_up: int = 2,
):
    r"""
    The T-GATE scheduler function 

    Args:
        cur_step (`int`):
            The current time step. 
        gate_step (`int` defaults to 10): 
            The time step to stop calculating the cross attention.
        sp_interval (`int` defaults to 5): 
            The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).
        fi_interval (`int` defaults to 1): 
            The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).
        warm_up (`int` defaults to 2): 
            The time step to warm up the model inference.

    Returns:
        ca_kward: (`dict`):
            A kwargs dictionary to pass along to the cross attention for caching and reusing. 
        sa_kward: (`dict`):
            A kwargs dictionary to pass along to the self attention for caching and reusing. 
        keep_shape (`bool`):
            Whether or not to remain the shape of hidden features
    """
    if cur_step < gate_step-1:
        # Semantics-Planning Stage
        ca_kwards = {
            'cache': False,
            'reuse': False,
        }
        if cur_step < warm_up:
            sa_kwards = {
                'cache': False,
                'reuse': False,
            }
        elif cur_step == warm_up:
            sa_kwards = {
                'cache': True,
                'reuse': False,
            }   
        else:
            if cur_step % sp_interval == 0:
                sa_kwards = {
                    'cache': True,
                    'reuse': False,
                }
            else:
                sa_kwards = {
                    'cache': False,
                    'reuse': True,
                }
        keep_shape = True
    
    elif cur_step == gate_step-1:
        ca_kwards = {
            'cache': True,
            'reuse': False,
        }
        sa_kwards = {
            'cache':True,
            'reuse':False
        }
        keep_shape = False
    else:
        # Fidelity-Improving Stage
        ca_kwards = {
            'cache': False,
            'reuse': True,
        }
        if cur_step % fi_interval == 0:
            sa_kwards = {
                'cache':True,
                'reuse':False
            }
        else:
            sa_kwards = {
                'cache':False,
                'reuse':True
            }
        keep_shape = True
    return ca_kwards, sa_kwards, keep_shape

def apply_rotary_emb(
    x: paddle.Tensor,
    freqs_cis: Union[paddle.Tensor, Tuple[paddle.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape([*tuple(x.shape)[:-1], -1, 2]).unbind(axis=-1)
            x_rotated = paddle.stack(x=[-x_imag, x_real], axis=-1).flatten(start_axis=3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape([*tuple(x.shape)[:-1], 2, -1]).unbind(axis=-2)
            x_rotated = paddle.concat(x=[-x_imag, x_real], axis=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        out = (x.astype(dtype="float32") * cos + x_rotated.astype(dtype="float32") * sin).astype(x.dtype)
        return out
    else:
        x_rotated = paddle.as_complex(x=x.astype(dtype="float32").reshape(*tuple(x.shape)[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(axis=2)
        x_out = paddle.as_real(x=x_rotated * freqs_cis).flatten(start_axis=3)
        return x_out.astype(dtype=x.dtype)