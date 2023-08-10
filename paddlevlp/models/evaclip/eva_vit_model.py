import paddle
import math
import os
from typing import Union, Dict
from functools import partial
from .eva_text_model import PatchDropout, AttentionalPooler, LayerNorm
from .modules.rope import VisionRotaryEmbeddingFast
from .modules.fusedln import FusedLayerNorm
from .utils import to_2tuple, trunc_normal_

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.log import logger


def drop_path(x,
              drop_prob: float=0.0,
              training: bool=False,
              scale_by_keep: bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    bern_0 = paddle.to_tensor(
        [keep_prob], dtype=paddle.float32) if not isinstance(
            keep_prob, paddle.Tensor) else keep_prob
    random_tensor = paddle.assign(
        paddle.bernoulli(
            paddle.broadcast_to(
                bern_0, paddle.empty(
                    shape=shape, dtype=x.dtype).shape)),
        paddle.empty(
            shape=shape, dtype=x.dtype))
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor.divide(keep_prob)
    return x * random_tensor


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        with get_rng_state_tracker().rng_state("global_seed"):
            out = drop_path(x, self.drop_prob, self.training)
        return out

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(paddle.nn.Layer):
    def __init__(self,
                 config,
                 act_layer=paddle.nn.GELU,
                 norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        out_features = in_features
        hidden_features = hidden_features or in_features
        if dist.get_world_size() > 1:
            self.fc1 = fleet.meta_parallel.ColumnParallelLinear(
                in_features,
                hidden_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True)
            self.fc2 = fleet.meta_parallel.ColumnParallelLinear(
                hidden_features,
                out_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True)
        else:
            self.fc1 = paddle.nn.Linear(in_features, hidden_features)
            self.fc2 = paddle.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(
            hidden_features) if config.subln else paddle.nn.Identity()
        self.drop = paddle.nn.Dropout(p=config.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.drop(x)
        return x


class SwiGLU(paddle.nn.Layer):
    def __init__(
            self,
            config,
            drop=0.0,
            act_layer=paddle.nn.Silu,
            norm_layer=paddle.nn.LayerNorm, ):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        out_features = in_features
        if dist.get_world_size() > 1:
            self.w1 = fleet.meta_parallel.ColumnParallelLinear(
                in_features,
                hidden_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True)
            self.w2 = fleet.meta_parallel.ColumnParallelLinear(
                in_features,
                hidden_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True)
            self.w3 = fleet.meta_parallel.ColumnParallelLinear(
                hidden_features,
                out_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True)
        else:
            self.w1 = paddle.nn.Linear(in_features, hidden_features)
            self.w2 = paddle.nn.Linear(in_features, hidden_features)
            self.w3 = paddle.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(
            hidden_features) if config.subln else paddle.nn.Identity()
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    def __init__(self,
                 config,
                 window_size=None,
                 rope=None,
                 norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        dim = config.embed_dim
        self.xattn_drop = config.attn_drop_rate
        self.xattn = config.xattn
        self.subln = config.subln

        self.num_heads = config.embed_dim // config.head_width
        head_dim = dim // self.num_heads
        if hasattr(config,
                   "attn_head_dim") and config.attn_head_dim is not None:
            head_dim = config.attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = config.qk_scale or head_dim**-0.5
        if self.subln:
            if dist.get_world_size() > 1:
                self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                    dim,
                    all_head_dim,
                    weight_attr=None,
                    has_bias=config.qkv_bias,
                    gather_output=True)
                self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                    dim,
                    all_head_dim,
                    weight_attr=None,
                    has_bias=False,
                    gather_output=True)
                self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                    dim,
                    all_head_dim,
                    weight_attr=None,
                    has_bias=config.qkv_bias,
                    gather_output=True)
            else:
                self.q_proj = paddle.nn.Linear(dim, all_head_dim)
                self.k_proj = paddle.nn.Linear(
                    dim, all_head_dim, bias_attr=False)
                self.v_proj = paddle.nn.Linear(dim, all_head_dim)
        else:
            if dist.get_world_size() > 1:
                self.qkv = fleet.meta_parallel.ColumnParallelLinear(
                    dim,
                    all_head_dim * 3,
                    weight_attr=None,
                    has_bias=False,
                    gather_output=True)
            else:
                self.qkv = paddle.nn.Linear(
                    dim, all_head_dim * 3, bias_attr=False)
            if config.qkv_bias:
                mpsize = 1
                if dist.get_world_size() > 1:
                    mpsize = fleet.get_hybrid_communicate_group(
                    ).get_model_parallel_world_size()
                init_data = paddle.zeros(shape=[all_head_dim // mpsize])
                self.q_bias = self.create_parameter(
                    shape=[all_head_dim // mpsize],
                    default_initializer=paddle.nn.initializer.Assign(init_data))
                self.v_bias = self.create_parameter(
                    shape=[all_head_dim // mpsize],
                    default_initializer=paddle.nn.initializer.Assign(init_data))
            else:
                self.q_bias = None
                self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1) + 3
            init_data = paddle.zeros(
                shape=[self.num_relative_distance, self.num_heads])
            self.relative_position_bias_table = self.create_parameter(
                shape=[self.num_relative_distance, self.num_heads],
                default_initializer=paddle.nn.initializer.Assign(init_data))
            coords_h = paddle.arange(end=window_size[0])
            coords_w = paddle.arange(end=window_size[1])
            coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = coords_flatten[:, :, (None)] - coords_flatten[:, (
                None), :]
            relative_coords = relative_coords.transpose(perm=[1, 2, 0])
            relative_coords[:, :, (0)] += window_size[0] - 1
            relative_coords[:, :, (1)] += window_size[1] - 1
            relative_coords[:, :, (0)] *= 2 * window_size[1] - 1
            relative_position_index = paddle.zeros(
                shape=(window_size[0] * window_size[1] + 1, ) * 2,
                dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
            relative_position_index[(0), 0:] = self.num_relative_distance - 3
            relative_position_index[0:, (0)] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer('relative_position_index',
                                 relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = paddle.nn.Dropout(p=self.xattn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if (
            config.subln and config.inner_attn_ln) else paddle.nn.Identity()
        if dist.get_world_size() > 1:
            self.proj = fleet.meta_parallel.ColumnParallelLinear(
                all_head_dim,
                dim,
                weight_attr=None,
                has_bias=True,
                gather_output=True)
        else:
            self.proj = paddle.nn.Linear(all_head_dim, dim)
        self.proj_drop = paddle.nn.Dropout(p=config.drop_rate)
        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.reshape(
                (B, N, self.num_heads, -1)).transpose(perm=[0, 2, 1, 3])
            k = k.reshape(
                (B, N, self.num_heads, -1)).transpose(perm=[0, 2, 1, 3])
            v = v.reshape(
                (B, N, self.num_heads, -1)).transpose(perm=[0, 2, 1, 3])
        else:
            qkv_bias = None
            if self.q_bias is not None:
                out_0 = paddle.zeros_like(x=self.v_bias)
                out_0.stop_gradient = not False
                qkv_bias = paddle.concat(x=(self.q_bias, out_0, self.v_bias))
            qkv = paddle.nn.functional.linear(
                x=x, weight=self.qkv.weight, bias=qkv_bias)

            if dist.get_world_size() > 1:
                hcg = fleet.get_hybrid_communicate_group()
                if hcg.get_model_parallel_world_size() > 1:
                    model_parallel_group = hcg.get_model_parallel_group()
                    qkv = paddle.distributed.collective._c_concat(
                        qkv, group=model_parallel_group)

            qkv = qkv.reshape(
                (B, N, 3, self.num_heads, -1)).transpose(perm=[2, 0, 3, 1, 4])
            q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope:
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = paddle.concat(
                x=(q[:, :, :1, :], ro_q_t), axis=-2).astype(dtype=v.dtype)
            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = paddle.concat(
                x=(k[:, :, :1, :], ro_k_t), axis=-2).astype(dtype=v.dtype)
        if self.xattn:
            q = q.transpose(perm=[0, 2, 1, 3])
            k = k.transpose(perm=[0, 2, 1, 3])
            v = v.transpose(perm=[0, 2, 1, 3])
            x = memory_efficient_attention(
                q, k, v, p=self.xattn_drop, scale=self.scale)
            x = x.reshape((B, N, -1))
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.proj_drop(x)
        else:
            q = q * self.scale
            x = k
            perm_0 = list(range(x.ndim))
            perm_0[-2] = x.ndim - 1
            perm_0[-1] = x.ndim - 2
            attn = q @x.transpose(perm=perm_0)
            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.reshape((-1))].reshape(
                        (self.window_size[0] * self.window_size[1] + 1,
                         self.window_size[0] * self.window_size[1] + 1, -1))
                relative_position_bias = relative_position_bias.transpose(
                    perm=[2, 0, 1])
                attn = attn + relative_position_bias.unsqueeze(axis=0).astype(
                    dtype=attn.dtype)
            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.astype(dtype=attn.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.astype(dtype='bool')
                attn = paddle.where(~attn_mask[:, (None), (None), :], attn,
                                    float('-inf'))
            attn = paddle.nn.functional.softmax(attn, axis=-1)
            with get_rng_state_tracker().rng_state("global_seed"):
                attn = self.attn_drop(attn)
            x = attn @v
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            x = x.transpose(perm=perm_1).reshape((B, N, -1))
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.proj_drop(x)
        return x


class Block(paddle.nn.Layer):
    def __init__(self,
                 config,
                 drop_path=0.0,
                 window_size=None,
                 rope=None,
                 act_layer=paddle.nn.GELU,
                 norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        dim = config.embed_dim
        init_values = config.init_values
        self.postnorm = config.postnorm

        self.norm1 = norm_layer(dim)
        self.attn = Attention(config, window_size=window_size, rope=rope)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.norm2 = norm_layer(dim)
        if config.naiveswiglu:
            self.mlp = SwiGLU(config, norm_layer=norm_layer)
        else:
            self.mlp = Mlp(config, act_layer=act_layer)
        if init_values is not None and init_values > 0:
            init_data = init_values * paddle.ones(shape=dim)
            self.gamma_1 = self.create_parameter(
                shape=dim,
                default_initializer=paddle.nn.initializer.Assign(init_data))
            init_data = init_values * paddle.ones(shape=dim)
            self.gamma_2 = self.create_parameter(
                shape=dim,
                default_initializer=paddle.nn.initializer.Assign(init_data))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(
                        self.attn(
                            x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x),
                        rel_pos_bias=rel_pos_bias,
                        attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.postnorm:
            x = x + self.drop_path(self.gamma_1 * self.norm1(
                self.attn(
                    x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(paddle.nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, config):
        super().__init__()
        img_size = to_2tuple(config.img_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] //
                                                      patch_size[0])
        self.patch_shape = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = paddle.nn.Conv2D(
            in_channels=config.in_chans,
            out_channels=config.embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_axis=2)
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        x = x.transpose(perm=perm_2)
        return x


class RelativePositionBias(paddle.nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + 3
        init_data = paddle.zeros(shape=[self.num_relative_distance, num_heads])
        self.relative_position_bias_table = self.create_parameter(
            shape=[self.num_relative_distance, num_heads],
            default_initializer=paddle.nn.initializer.Assign(init_data))
        coords_h = paddle.arange(end=window_size[0])
        coords_w = paddle.arange(end=window_size[1])
        coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
        coords_flatten = paddle.flatten(x=coords, start_axis=1)
        relative_coords = coords_flatten[:, :, (None)] - coords_flatten[:, (
            None), :]
        relative_coords = relative_coords.transpose(perm=[1, 2, 0])
        relative_coords[:, :, (0)] += window_size[0] - 1
        relative_coords[:, :, (1)] += window_size[1] - 1
        relative_coords[:, :, (0)] *= 2 * window_size[1] - 1
        relative_position_index = paddle.zeros(
            shape=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
        relative_position_index[(0), 0:] = self.num_relative_distance - 3
        relative_position_index[0:, (0)] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape((-1))].reshape(
                (self.window_size[0] * self.window_size[1] + 1,
                 self.window_size[0] * self.window_size[1] + 1, -1))
        return relative_position_bias.transpose(perm=[2, 0, 1])


class EVAVisionTransformerConfig(PretrainedConfig):

    model_type = "evavision_transformer"
    attribute_map: Dict[str, str] = {}

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_values=None,
            patch_dropout=0.0,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            rope=False,
            use_mean_pooling=True,
            attentional_pool=False,
            n_queries: int=256,
            attn_pooler_heads: int=8,
            init_scale=0.001,
            enable_recompute=False,
            xattn=False,
            postnorm=False,
            pt_hw_seq_len=16,
            intp_freq=False,
            naiveswiglu=False,
            subln=False,
            output_tokens=False,
            fusedLN=False,
            inner_attn_ln=False,
            **kwargs, ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_values = init_values
        self.patch_dropout = patch_dropout
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.rope = rope
        self.use_mean_pooling = use_mean_pooling
        self.attentional_pool = attentional_pool
        self.n_queries = n_queries
        self.attn_pooler_heads = attn_pooler_heads
        self.init_scale = init_scale
        self.enable_recompute = enable_recompute
        self.xattn = xattn
        self.postnorm = postnorm
        self.pt_hw_seq_len = pt_hw_seq_len
        self.intp_freq = intp_freq
        self.naiveswiglu = naiveswiglu
        self.subln = subln
        self.output_tokens = output_tokens
        self.fusedLN = fusedLN
        self.inner_attn_ln = inner_attn_ln

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: Union[str, os.PathLike],
                        **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path,
                                                  **kwargs)

        if ("model_type" in config_dict and hasattr(cls, "model_type") and
                config_dict["model_type"] != cls.model_type):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        if "vision_cfg" in config_dict:
            config_dict = config_dict["vision_cfg"]

        return cls.from_dict(config_dict, **kwargs)


class EVAVisionTransformerPretrainedModel(PretrainedModel):
    """
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVAVisionTransformerConfig
    resource_files_names = {"model_state": "model_state_vision.pdparams"}
    base_model_prefix = "evavision_transformer"


class EVAVisionTransformer(EVAVisionTransformerPretrainedModel):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, config: EVAVisionTransformerConfig):
        super(EVAVisionTransformer, self).__init__(config)
        self.image_size = config.img_size
        self.enable_recompute = config.enable_recompute
        self.output_tokens = config.output_tokens
        self.num_classes = num_classes = config.num_classes
        self.embed_dim = embed_dim = config.embed_dim
        self.naiveswiglu = config.naiveswiglu
        use_mean_pooling = config.use_mean_pooling
        norm_layer = partial(
            FusedLayerNorm, epsilon=1e-6) if config.fusedLN else partial(
                LayerNorm, epsilon=1e-6)
        num_heads = config.embed_dim // config.head_width

        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        init_data = paddle.zeros(shape=[1, 1, embed_dim])
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim],
            default_initializer=paddle.nn.initializer.Assign(init_data))
        if config.use_abs_pos_emb:
            init_data = paddle.zeros(shape=[1, num_patches + 1, embed_dim])
            self.pos_embed = self.create_parameter(
                shape=[1, num_patches + 1, embed_dim],
                default_initializer=paddle.nn.initializer.Assign(init_data))
        else:
            self.pos_embed = None
        self.pos_drop = paddle.nn.Dropout(p=config.drop_rate)
        if config.use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        if config.rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = config.img_size // config.patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=config.pt_hw_seq_len,
                ft_seq_len=hw_seq_len if config.intp_freq else None)
        else:
            self.rope = None
        dpr = [
            x.item()
            for x in paddle.linspace(
                start=0, stop=config.drop_path_rate, num=config.depth)
        ]
        self.blocks = paddle.nn.LayerList(sublayers=[
            Block(
                config,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=self.patch_embed.patch_shape
                if config.use_rel_pos_bias else None,
                rope=self.rope) for i in range(config.depth)
        ])
        if config.attentional_pool:
            self.attn_pool = AttentionalPooler(config)
            self.norm = paddle.nn.Identity(
            ) if use_mean_pooling else norm_layer(num_classes)
            self.fc_norm = norm_layer(num_classes) if use_mean_pooling else None
            if dist.get_world_size() > 1:
                self.head = fleet.meta_parallel.ColumnParallelLinear(
                    num_classes,
                    num_classes,
                    weight_attr=None,
                    has_bias=True,
                    gather_output=True
                ) if num_classes > 0 else paddle.nn.Identity()
            else:
                self.head = paddle.nn.Linear(
                    num_classes,
                    num_classes) if num_classes > 0 else paddle.nn.Identity()
        else:
            self.attn_pool = None
            self.norm = paddle.nn.Identity(
            ) if use_mean_pooling else norm_layer(embed_dim)
            self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
            if dist.get_world_size() > 1:
                self.head = fleet.meta_parallel.ColumnParallelLinear(
                    embed_dim,
                    num_classes,
                    weight_attr=None,
                    has_bias=True,
                    gather_output=True
                ) if num_classes > 0 else paddle.nn.Identity()
            else:
                self.head = paddle.nn.Linear(
                    embed_dim,
                    num_classes) if num_classes > 0 else paddle.nn.Identity()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.head, fleet.meta_parallel.ColumnParallelLinear):
            trunc_normal_(self.head.weight, std=0.02)
            with paddle.no_grad():
                self.head.weight.set_value(
                    self.head.weight.scale(scale=config.init_scale))
                self.head.bias.set_value(
                    self.head.bias.scale(scale=config.init_scale))
        self.patch_dropout = PatchDropout(
            config.patch_dropout
        ) if config.patch_dropout > 0.0 else paddle.nn.Identity()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param = param.divide(paddle.to_tensor(math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight, layer_id + 1)

    def get_cast_dtype(self) -> paddle.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        zeros_params = paddle.nn.initializer.Constant(0.0)
        ones_params = paddle.nn.initializer.Constant(1.0)
        if isinstance(m, (paddle.nn.Linear,
                          fleet.meta_parallel.ColumnParallelLinear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                zeros_params(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            zeros_params(m.bias)
            ones_params(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.stop_gradient = not False

    def set_grad_checkpointing(self, enable=True):
        self.enable_recompute = enable

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if dist.get_world_size() > 1:
            self.head = fleet.meta_parallel.ColumnParallelLinear(
                self.embed_dim,
                num_classes,
                weight_attr=None,
                has_bias=True,
                gather_output=True) if num_classes > 0 else paddle.nn.Identity()
        else:
            self.head = paddle.nn.Linear(
                self.embed_dim,
                num_classes) if num_classes > 0 else paddle.nn.Identity()

    def forward_features(self, x, return_all_features=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand(shape=[batch_size, -1, -1])
        x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.pos_drop(x)
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout,
                                                paddle.nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(
                    self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(
                    self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)
        rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None
        cnt = 0
        for blk in self.blocks:
            cnt += 1
            if self.enable_recompute:
                x = paddle.distributed.fleet.utils.recompute(
                    blk, x, rel_pos_bias, use_reentrant=False)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if self.attn_pool is not None:
            x = self.attn_pool(x)
        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                if self.output_tokens:
                    return self.fc_norm(x.mean(axis=1)), x
                return self.fc_norm(x.mean(axis=1))
            else:
                if self.output_tokens:
                    return x[:, 0], x[:, 1:]
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        if self.output_tokens:
            x, tokens = self.forward_features(x)
            x = self.head(x)
            return x, tokens
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x
