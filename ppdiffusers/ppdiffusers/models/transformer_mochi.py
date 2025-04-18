import paddle
import paddle.nn as nn

from typing import Optional, Tuple

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from ..utils.paddle_utils import maybe_allow_in_graph
from .attention import FeedForward
from .attention_processor import MochiAttention, MochiAttnProcessor2_0
from .embeddings import MochiCombinedTimestepCaptionEmbedding, PatchEmbed
from .modeling_outputs import Transformer2DModelOutput
from .modeling_utils import ModelMixin
from .normalization import AdaLayerNormContinuous, RMSNorm, MochiRMSNorm

logger = logging.get_logger(__name__)

class MochiModulatedRMSNorm(nn.Layer):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps
        self.norm = MochiRMSNorm(0, eps, False)

    def forward(self, hidden_states, scale=None):
        hidden_states_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype('float32')
        hidden_states = self.norm(hidden_states)
        if scale is not None:
            hidden_states = hidden_states * scale
        hidden_states = hidden_states.astype(hidden_states_dtype)
        return hidden_states


class MochiLayerNormContinuous(nn.Layer):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps=1e-5,
        bias=True,
    ):
        super().__init__()
        self.silu = nn.Silu()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias_attr=bias)
        self.norm = MochiModulatedRMSNorm(eps=eps)

    def forward(
        self,
        x: paddle.Tensor,
        conditioning_embedding: paddle.Tensor,
    ) -> paddle.Tensor:
        input_dtype = x.dtype
        scale = self.linear_1(self.silu(conditioning_embedding).astype(x.dtype))
        x = self.norm(x, (1 + scale.unsqueeze(1).astype('float32')))
        return x.astype(input_dtype)

class MochiRMSNormZero(nn.Layer):
    def __init__(
        self, embedding_dim: int, hidden_dim: int, eps: float = 1e-5, elementwise_affine: bool = False
    ) -> None:
        super().__init__()
        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.norm = MochiRMSNorm(0, eps, False)

    def forward(
        self, hidden_states: paddle.Tensor, emb: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        hidden_states_dtype = hidden_states.dtype
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = paddle.split(emb, num_or_sections=4, axis=1)
        hidden_states = self.norm(hidden_states.astype('float32')) * (1 + scale_msa[:, None].astype('float32'))
        hidden_states = hidden_states.astype(hidden_states_dtype)
        return hidden_states, gate_msa, scale_mlp, gate_mlp

@maybe_allow_in_graph
class MochiTransformerBlock(nn.Layer):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        qk_norm: str = "rms_norm",
        activation_fn: str = "swiglu",
        context_pre_only: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.context_pre_only = context_pre_only
        self.ff_inner_dim = (4 * dim * 2) // 3
        self.ff_context_inner_dim = (4 * pooled_projection_dim * 2) // 3
        self.norm1 = MochiRMSNormZero(dim, 4 * dim, eps=eps, elementwise_affine=False)
        if not context_pre_only:
            self.norm1_context = MochiRMSNormZero(dim, 4 * pooled_projection_dim, eps=eps, elementwise_affine=False)
        else:
            self.norm1_context = MochiLayerNormContinuous(
                embedding_dim=pooled_projection_dim,
                conditioning_embedding_dim=dim,
                eps=eps,
            )
        self.attn1 = MochiAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
            added_kv_proj_dim=pooled_projection_dim,
            added_proj_bias=False,
            out_dim=dim,
            out_context_dim=pooled_projection_dim,
            context_pre_only=context_pre_only,
            processor=MochiAttnProcessor2_0(),
            eps=1e-5,
        )
        self.norm2 = MochiModulatedRMSNorm(eps=eps)
        self.norm2_context = MochiModulatedRMSNorm(eps=eps) if not self.context_pre_only else None
        self.norm3 = MochiModulatedRMSNorm(eps)
        self.norm3_context = MochiModulatedRMSNorm(eps=eps) if not self.context_pre_only else None
        self.ff = FeedForward(dim, inner_dim=self.ff_inner_dim, activation_fn=activation_fn, bias=False)
        self.ff_context = None
        if not context_pre_only:
            self.ff_context = FeedForward(
                pooled_projection_dim,
                inner_dim=self.ff_context_inner_dim,
                activation_fn=activation_fn,
                bias=False,
            )
        self.norm4 = MochiModulatedRMSNorm(eps=eps)
        self.norm4_context = MochiModulatedRMSNorm(eps=eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        encoder_attention_mask: paddle.Tensor,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        if not self.context_pre_only:
            norm_encoder_hidden_states, enc_gate_msa, enc_scale_mlp, enc_gate_mlp = self.norm1_context(
                encoder_hidden_states, temb
            )
        else:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        attn_hidden_states, context_attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + self.norm2(attn_hidden_states, paddle.tanh(gate_msa).unsqueeze(1))
        norm_hidden_states = self.norm3(hidden_states, (1 + scale_mlp.unsqueeze(1).astype('float32')))
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + self.norm4(ff_output, paddle.tanh(gate_mlp).unsqueeze(1))
        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + self.norm2_context(
                context_attn_hidden_states, paddle.tanh(enc_gate_msa).unsqueeze(1)
            )
            norm_encoder_hidden_states = self.norm3_context(
                encoder_hidden_states, (1 + enc_scale_mlp.unsqueeze(1).astype('float32'))
            )
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + self.norm4_context(
                context_ff_output, paddle.tanh(enc_gate_mlp).unsqueeze(1)
            )
        return hidden_states, encoder_hidden_states

class MochiRoPE(nn.Layer):
    def __init__(self, base_height: int = 192, base_width: int = 192) -> None:
        super().__init__()
        self.target_area = base_height * base_width

    def _centers(self, start, stop, num, dtype) -> paddle.Tensor:
        edges = paddle.linspace(start, stop, num + 1, dtype=dtype)
        return (edges[:-1] + edges[1:]) / 2

    def _get_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        dtype: Optional[str] = None,
    ) -> paddle.Tensor:
        scale = (self.target_area / (height * width)) ** 0.5
        t = paddle.arange(num_frames, dtype=dtype)
        h = self._centers(-height * scale / 2, height * scale / 2, height, dtype)
        w = self._centers(-width * scale / 2, width * scale / 2, width, dtype)
        grid_t, grid_h, grid_w = paddle.meshgrid(t, h, w)
        positions = paddle.stack([grid_t, grid_h, grid_w], axis=-1).reshape([-1, 3])
        return positions

    def _create_rope(self, freqs: paddle.Tensor, pos: paddle.Tensor) -> paddle.Tensor:
        freqs = paddle.einsum('nd,dhf->nhf', pos.astype('float32'), freqs.astype('float32'))
        freqs_cos = paddle.cos(freqs)
        freqs_sin = paddle.sin(freqs)
        return freqs_cos, freqs_sin

    def forward(
        self,
        pos_frequencies: paddle.Tensor,
        num_frames: int,
        height: int,
        width: int,
        dtype: Optional[str] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        pos = self._get_positions(num_frames, height, width, dtype)
        rope_cos, rope_sin = self._create_rope(pos_frequencies, pos)
        return rope_cos, rope_sin

# @maybe_allow_in_graph
# class MochiTransformer3DModel(ModelMixin, ConfigMixin):
    
#     _supports_gradient_checkpointing = True
#     _no_split_modules = ["MochiTransformerBlock"]
#     _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

#     @register_to_config
#     def __init__(
#         self,
#         patch_size: int = 2,
#         num_attention_heads: int = 24,
#         attention_head_dim: int = 128,
#         num_layers: int = 48,
#         pooled_projection_dim: int = 1536,
#         in_channels: int = 12,
#         out_channels: Optional[int] = None,
#         qk_norm: str = "rms_norm",
#         text_embed_dim: int = 4096,
#         time_embed_dim: int = 256,
#         activation_fn: str = "swiglu",
#         max_sequence_length: int = 256,
#     ) -> None:
#         super().__init__()
#         inner_dim = num_attention_heads * attention_head_dim
#         out_channels = out_channels or in_channels
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size,
#             in_channels=in_channels,
#             embed_dim=inner_dim,
#         )
#         self.time_embed = MochiCombinedTimestepCaptionEmbedding(
#             embedding_dim=inner_dim,
#             pooled_projection_dim=pooled_projection_dim,
#             text_embed_dim=text_embed_dim,
#             time_embed_dim=time_embed_dim,
#             num_attention_heads=8,
#         )
#         self.pos_frequencies = self.create_parameter(
#             shape=[3, num_attention_heads, attention_head_dim // 2],
#             default_initializer=nn.initializer.Constant(value=0.0)
#         )
#         self.rope = MochiRoPE()
#         self.transformer_blocks = nn.LayerList(
#             [
#                 MochiTransformerBlock(
#                     dim=inner_dim,
#                     num_attention_heads=num_attention_heads,
#                     attention_head_dim=attention_head_dim,
#                     pooled_projection_dim=pooled_projection_dim,
#                     qk_norm=qk_norm,
#                     activation_fn=activation_fn,
#                     context_pre_only=i == num_layers - 1,
#                 )
#                 for i in range(num_layers)
#             ]
#         )
#         self.norm_out = AdaLayerNormContinuous(
#             inner_dim,
#             inner_dim,
#             elementwise_affine=False,
#             eps=1e-6,
#             norm_type="layer_norm",
#         )
#         self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)
#         self.gradient_checkpointing = False

#     def forward(
#         self,
#         hidden_states: paddle.Tensor,
#         encoder_hidden_states: paddle.Tensor,
#         timestep: paddle.Tensor,
#         encoder_attention_mask: paddle.Tensor,
#         return_dict: bool = True,
#     ) -> paddle.Tensor:

#         batch_size, num_channels, num_frames, height, width = hidden_states.shape
#         p = self.config.patch_size
#         post_patch_height = height // p
#         post_patch_width = width // p
#         temb, encoder_hidden_states = self.time_embed(
#             timestep,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             hidden_dtype=hidden_states.dtype,
#         )
#         hidden_states = hidden_states.transpose([0, 2, 1, 3, 4]).flatten(0, 1)
#         hidden_states = self.patch_embed(hidden_states)
#         hidden_states = hidden_states.unflatten(0, [batch_size, -1]).flatten(1, 2)
#         image_rotary_emb = self.rope(
#             self.pos_frequencies,
#             num_frames,
#             post_patch_height,
#             post_patch_width,
#             dtype='float32',
#         )
#         for block in self.transformer_blocks:
#             if self.training and self.gradient_checkpointing:
#                 def create_custom_forward(module):
#                     def create_forward(*inputs):
#                         return module(*inputs)

#                     return create_forward

#                 hidden_states,encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
#                     create_custom_forward(block), 
#                     hidden_states=hidden_states,
#                     encoder_hidden_states=encoder_hidden_states,
#                     temb=temb,
#                     encoder_attention_mask=encoder_attention_mask,
#                     image_rotary_emb=image_rotary_emb,
#                 )
#             else:
#                 hidden_states,encoder_hidden_states = block(
#                     hidden_states=hidden_states,
#                     encoder_hidden_states=encoder_hidden_states,
#                     temb=temb,
#                     encoder_attention_mask=encoder_attention_mask,
#                     image_rotary_emb=image_rotary_emb,
#                 )
                
#         hidden_states = self.norm_out(hidden_states, temb)
#         hidden_states = self.proj_out(hidden_states)
#         hidden_states = hidden_states.reshape([batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1])
#         hidden_states = hidden_states.transpose([0, 6, 1, 2, 4, 3, 5])
#         output = hidden_states.reshape([batch_size, -1, num_frames, height, width])
#         if not return_dict:
#             return (output,)
#         return Transformer2DModelOutput(sample=output)


@maybe_allow_in_graph
class MochiTransformer3DModel(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True
    _no_split_modules = ["MochiTransformerBlock"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        pooled_projection_dim: int = 1536,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
    ) -> None:
        print("\n==== MochiTransformer3DModel Initialization Start ====")
        print(f"Parameters: patch_size={patch_size}, num_attention_heads={num_attention_heads}, attention_head_dim={attention_head_dim}")
        print(f"Parameters: num_layers={num_layers}, pooled_projection_dim={pooled_projection_dim}")
        print(f"Parameters: in_channels={in_channels}, out_channels={out_channels}, qk_norm={qk_norm}")
        print(f"Parameters: text_embed_dim={text_embed_dim}, time_embed_dim={time_embed_dim}")
        print(f"Parameters: activation_fn={activation_fn}, max_sequence_length={max_sequence_length}")
        
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        print(f"Calculated inner_dim={inner_dim}")
        
        out_channels = out_channels or in_channels
        print(f"Using out_channels={out_channels}")
        
        print("Initializing patch_embed...")
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
        )
        
        print("Initializing time_embed...")
        self.time_embed = MochiCombinedTimestepCaptionEmbedding(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            text_embed_dim=text_embed_dim,
            time_embed_dim=time_embed_dim,
            num_attention_heads=8,
        )
        
        print("Creating pos_frequencies parameter...")
        self.pos_frequencies = self.create_parameter(
            shape=[3, num_attention_heads, attention_head_dim // 2],
            default_initializer=nn.initializer.Constant(value=0.0)
        )
        print(f"pos_frequencies shape: {self.pos_frequencies.shape}")
        
        print("Initializing rope...")
        self.rope = MochiRoPE()
        
        print(f"Creating {num_layers} transformer blocks...")
        self.transformer_blocks = nn.LayerList(
            [
                MochiTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    pooled_projection_dim=pooled_projection_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    context_pre_only=i == num_layers - 1,
                )
                for i in range(num_layers)
            ]
        )
        print(f"Created {len(self.transformer_blocks)} transformer blocks")
        
        print("Initializing norm_out...")
        self.norm_out = AdaLayerNormContinuous(
            inner_dim,
            inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            norm_type="layer_norm",
        )
        
        print("Initializing proj_out...")
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)
        print(f"proj_out: in_features={inner_dim}, out_features={patch_size * patch_size * out_channels}")
        
        self.gradient_checkpointing = False
        print("Gradient checkpointing set to False")
        print("==== MochiTransformer3DModel Initialization Complete ====\n")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        encoder_attention_mask: paddle.Tensor,
        return_dict: bool = True,
    ) -> paddle.Tensor:
        print(f"\n==== MochiTransformer3DModel Forward Start ====")
        print(f"Input hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
        print(f"Input encoder_hidden_states: shape={encoder_hidden_states.shape}, dtype={encoder_hidden_states.dtype}")
        print(f"Input timestep: shape={timestep.shape}, dtype={timestep.dtype}, value={timestep}")
        print(f"Input encoder_attention_mask: shape={encoder_attention_mask.shape}, dtype={encoder_attention_mask.dtype}")
        
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        print(f"Extracted dimensions: batch_size={batch_size}, num_channels={num_channels}, num_frames={num_frames}, height={height}, width={width}")
        
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p
        print(f"Patch size={p}, post_patch_height={post_patch_height}, post_patch_width={post_patch_width}")
        
        # Time embedding
        print(f"Before time_embed - encoder_hidden_states shape: {encoder_hidden_states.shape}")
        temb, encoder_hidden_states = self.time_embed(
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_dtype=hidden_states.dtype,
        )
        print(f"After time_embed - temb: shape={temb.shape}, dtype={temb.dtype}, mean={temb.mean().item()}, max={temb.max().item()}, min={temb.min().item()}")
        print(f"After time_embed - encoder_hidden_states: shape={encoder_hidden_states.shape}, dtype={encoder_hidden_states.dtype}")
        
        # Reshape for patch embedding
        print(f"Before reshape - hidden_states: shape={hidden_states.shape}")
        hidden_states = hidden_states.transpose([0, 2, 1, 3, 4]).flatten(0, 1)
        print(f"After transpose and flatten - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}")
        
        # Patch embedding
        hidden_states = self.patch_embed(hidden_states)
        print(f"After patch_embed - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}, max={hidden_states.max().item()}, min={hidden_states.min().item()}")
        
        # Reshape for transformer
        hidden_states = hidden_states.unflatten(0, [batch_size, -1]).flatten(1, 2)
        print(f"After unflatten and flatten - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}")
        
        # Rotary position embedding
        print(f"pos_frequencies shape: {self.pos_frequencies.shape}, dtype: {self.pos_frequencies.dtype}")
        image_rotary_emb = self.rope(
            self.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            dtype='float32',
        )
        print(f"image_rotary_emb: type={type(image_rotary_emb)}, length={len(image_rotary_emb)}")
        if isinstance(image_rotary_emb, (list, tuple)):
            for i, emb in enumerate(image_rotary_emb):
                print(f"image_rotary_emb[{i}]: shape={emb.shape}, dtype={emb.dtype}, mean={emb.mean().item() if hasattr(emb, 'mean') else 'N/A'}")
        
        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            print(f"\nProcessing transformer block {i}/{len(self.transformer_blocks)-1}")
            print(f"Before block {i} - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}")
            print(f"Before block {i} - encoder_hidden_states: shape={encoder_hidden_states.shape}, mean={encoder_hidden_states.mean().item()}")
            
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)
                    return create_forward

                hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                    create_custom_forward(block), 
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            
            print(f"After block {i} - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}, max={hidden_states.max().item()}, min={hidden_states.min().item()}")
            print(f"After block {i} - encoder_hidden_states: shape={encoder_hidden_states.shape}, mean={encoder_hidden_states.mean().item()}")
        
        # Final normalization and projection
        print("\nStarting final norm and projection")
        print(f"Before norm_out - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}")
        hidden_states = self.norm_out(hidden_states, temb)
        print(f"After norm_out - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}")
        
        hidden_states = self.proj_out(hidden_states)
        print(f"After proj_out - hidden_states: shape={hidden_states.shape}, mean={hidden_states.mean().item()}, max={hidden_states.max().item()}, min={hidden_states.min().item()}")
        
        # Final reshape
        print(f"Before final reshape - hidden_states: shape={hidden_states.shape}")
        hidden_states = hidden_states.reshape([batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1])
        print(f"After reshape - hidden_states: shape={hidden_states.shape}")
        
        hidden_states = hidden_states.transpose([0, 6, 1, 2, 4, 3, 5])
        print(f"After transpose - hidden_states: shape={hidden_states.shape}")
        
        output = hidden_states.reshape([batch_size, -1, num_frames, height, width])
        print(f"Final output: shape={output.shape}, mean={output.mean().item()}, max={output.max().item()}, min={output.min().item()}")
        print("==== MochiTransformer3DModel Forward End ====\n")
        
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
