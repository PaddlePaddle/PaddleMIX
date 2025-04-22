import paddle
from typing import Any, Dict, Optional, Tuple, Union
from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from ..utils.paddle_utils import maybe_allow_in_graph
from .attention import Attention, FeedForward
from .attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from .cache_utils import CacheMixin
from .embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from .modeling_outputs import Transformer2DModelOutput
from .modeling_utils import ModelMixin
from .normalization import AdaLayerNorm, CogVideoXLayerNormZero
logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class CogVideoXBlock(paddle.nn.Layer):
    """
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int,
        attention_head_dim: int, 
        time_embed_dim: int, 
        dropout: float=0.0,
        activation_fn: str='gelu-approximate', 
        attention_bias: bool=False,
        qk_norm: bool=True, 
        norm_elementwise_affine: bool=True, 
        norm_eps: float=1e-05, 
        final_dropout: bool=True, 
        ff_inner_dim: Optional[int]=None, 
        ff_bias: bool=True, 
        attention_out_bias: bool=True
    ):
        super().__init__()
        
        # 1. self attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim, 
            dim_head=attention_head_dim,
            heads=num_attention_heads, 
            qk_norm='layer_norm' if qk_norm else None, 
            eps=1e-06, 
            bias=attention_bias, 
            out_bias=attention_out_bias, 
            processor=CogVideoXAttnProcessor2_0()
        )

        # 2. feed forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = FeedForward(
            dim, 
            dropout=dropout, 
            activation_fn=activation_fn, 
            final_dropout=final_dropout, 
            inner_dim=ff_inner_dim, 
            bias=ff_bias
        )

    def forward(
        self, 
        hidden_states: paddle.Tensor, 
        encoder_hidden_states:paddle.Tensor, 
        temb: paddle.Tensor, 
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]]=None
    ) ->paddle.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]

        # norm and modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states, 
            encoder_hidden_states=norm_encoder_hidden_states, 
            image_rotary_emb=image_rotary_emb
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm and modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed forward
        norm_hidden_states = paddle.concat([norm_encoder_hidden_states, norm_hidden_states], axis=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length])

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, num_attention_heads: int=30, attention_head_dim: int
        =64, in_channels: int=16, out_channels: Optional[int]=16,
        flip_sin_to_cos: bool=True, freq_shift: int=0, time_embed_dim: int=
        512, text_embed_dim: int=4096, num_layers: int=30, dropout: float=
        0.0, attention_bias: bool=True, sample_width: int=90, sample_height:
        int=60, sample_frames: int=49, patch_size: int=2,
        temporal_compression_ratio: int=4, max_text_seq_length: int=226,
        activation_fn: str='gelu-approximate', timestep_activation_fn: str=
        'silu', norm_elementwise_affine: bool=True, norm_eps: float=1e-05,
        spatial_interpolation_scale: float=1.875,
        temporal_interpolation_scale: float=1.0,
        use_rotary_positional_embeddings: bool=False,
        use_learned_positional_embeddings: bool=False):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        if (not use_rotary_positional_embeddings and
            use_learned_positional_embeddings):
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional embeddings. If you're using a custom model and/or believe this should be supported, please open an issue at https://github.com/huggingface/diffusers/issues."
                )
        self.patch_embed = CogVideoXPatchEmbed(patch_size=patch_size,
            in_channels=in_channels, embed_dim=inner_dim, text_embed_dim=
            text_embed_dim, bias=True, sample_width=sample_width,
            sample_height=sample_height, sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings
            )
        self.embedding_dropout = paddle.nn.Dropout(p=dropout)
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim,
            timestep_activation_fn)
        self.transformer_blocks = paddle.nn.LayerList(sublayers=[
            CogVideoXBlock(dim=inner_dim, num_attention_heads=
            num_attention_heads, attention_head_dim=attention_head_dim,
            time_embed_dim=time_embed_dim, dropout=dropout, activation_fn=
            activation_fn, attention_bias=attention_bias,
            norm_elementwise_affine=norm_elementwise_affine, norm_eps=
            norm_eps) for _ in range(num_layers)])
        self.norm_final = paddle.nn.LayerNorm(normalized_shape=inner_dim,
            epsilon=norm_eps, weight_attr=norm_elementwise_affine,
            bias_attr=norm_elementwise_affine)
        self.norm_out = AdaLayerNorm(embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim, norm_elementwise_affine=
            norm_elementwise_affine, norm_eps=norm_eps, chunk_dim=1)
        self.proj_out = paddle.nn.Linear(in_features=inner_dim,
            out_features=patch_size * patch_size * out_channels)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    def attn_processors(self) ->Dict[str, AttentionProcessor]:
        """
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        processors = {}

        def fn_recursive_add_processors(name: str, module: paddle.nn.Layer,
            processors: Dict[str, AttentionProcessor]):
            if hasattr(module, 'get_processor'):
                processors[f'{name}.processor'] = module.get_processor()
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f'{name}.{sub_name}', child,
                    processors)
            return processors
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[
        str, AttentionProcessor]]):
        """
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f'A dict of processors was passed, but the number of processors {len(processor)} does not match the number of attention layers: {count}. Please make sure to pass {count} processor classes.'
                )

        def fn_recursive_attn_processor(name: str, module: paddle.nn.Layer,
            processor):
            if hasattr(module, 'set_processor'):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f'{name}.processor'))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f'{name}.{sub_name}', child,
                    processor)
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None
        for _, attn_processor in self.attn_processors.items():
            if 'Added' in str(attn_processor.__class__.__name__):
                raise ValueError(
                    '`fuse_qkv_projections()` is not supported for models having added KV projections.'
                    )
        self.original_attn_processors = self.attn_processors
        for module in self.sublayers():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)
        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self, 
        hidden_states: paddle.Tensor, 
        encoder_hidden_states: paddle.Tensor, 
        timestep: Union[int, float, paddle.Tensor],
        timestep_cond: Optional[paddle.Tensor]=None, 
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]]=None, 
        return_dict: bool=True
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.cast(hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = tuple(encoder_hidden_states.shape)[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                raise NotImplementedError
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states, 
                    encoder_hidden_states=encoder_hidden_states, 
                    temb=emb, 
                    image_rotary_emb=image_rotary_emb
                )
            # print("hidden_states:", hidden_states.abs().mean().item(), hidden_states.min().item(), hidden_states.max().item())
            # print("encoder_hidden_states:", encoder_hidden_states.abs().mean().item(), encoder_hidden_states.min().item(), encoder_hidden_states.max().item())
            
        if not self.config.use_rotary_positional_embeddings:
            # 2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # 5B
            hidden_states = paddle.concat(x=[encoder_hidden_states,
                hidden_states], axis=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]
        
        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape([batch_size, num_frames, height // p, width // p, -1, p, p])
        output = output.transpose(perm=[0, 1, 4, 2, 5, 3, 6]).flatten(5, 6).flatten(3, 4)
        if not return_dict:
            return output,
        return Transformer2DModelOutput(sample=output)
