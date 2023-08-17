import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import os
import copy
from typing import Union, Dict
from functools import partial
from ..evaclip.modules.rope import VisionRotaryEmbeddingFast
from .modeling_finetune import Block, PatchEmbed, RelativePositionBias, DecoupledRelativePositionBias
try:
    from ..evaclip.modules.fusedln import FusedLayerNorm
except:
    from paddle.nn import LayerNorm as FusedLayerNorm
    print("Warning, FusedLn module is not available, use LayerNorm instead.")
from ..evaclip.utils import trunc_normal_

import paddle.distributed as dist
from IPython import embed
import numpy as np
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlevlp.models.evaclip.eva_clip_model import EVACLIP, EVACLIPConfig

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.log import logger


__all__ = [
    "EVA02VisionTransformerForMIMConfig",
    "EVA02VisionTransformerForMIMPretrainedModel",
    "EVA02VisionTransformerForMIM",
    "EVA02ForPretrainConfig",
    "EVA02ForPretrain",
]


class EVA02VisionTransformerForMIMConfig(PretrainedConfig):

    model_type = "eva02_vit_pretrain"
    attribute_map: Dict[str, str] = {}

    def __init__(
            self,
            img_size=224,
            patch_size=14,
            in_chans=3,
            # num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            init_values=None,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            use_shared_decoupled_rel_pos_bias=False,
            #use_mean_pooling=True,
            init_scale=0.001,
            enable_recompute=False,
            stop_grad_conv1=True, #
            postnorm=False,
            deepnorm=False, #
            subln=True,
            xattn=False,
            swiglu=False,
            naiveswiglu=False,
            rope=True,
            init_std=0.02, #
            xavier_normal_init=True, #
            attn_head_dim=None, #
            predict_feature_dim=768, #
            # pt_hw_seq_len=16,
            # intp_freq=False,
            fusedLN=False,
            **kwargs, ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        #self.num_classes = num_classes
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
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_shared_decoupled_rel_pos_bias = use_shared_decoupled_rel_pos_bias
        #self.use_mean_pooling = use_mean_pooling
        self.init_scale = init_scale
        self.enable_recompute = enable_recompute
        self.stop_grad_conv1 = stop_grad_conv1
        self.deepnorm = deepnorm
        self.postnorm = postnorm
        self.xattn = xattn
        #self.intp_freq = intp_freq
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu
        self.rope = rope
        #self.pt_hw_seq_len = pt_hw_seq_len
        self.subln = subln
        self.fusedLN = fusedLN

        self.init_std = init_std
        self.xavier_normal_init = xavier_normal_init
        self.attn_head_dim = attn_head_dim
        self.predict_feature_dim = predict_feature_dim

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

        return cls.from_dict(config_dict, **kwargs)


class EVA02VisionTransformerForMIMPretrainedModel(PretrainedModel):
    """
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVA02VisionTransformerForMIMConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "eva02_vit_pretrain"


class EVA02VisionTransformerForMIM(EVA02VisionTransformerForMIMPretrainedModel):

    def __init__(self, config: EVA02VisionTransformerForMIMConfig):
        super(EVA02VisionTransformerForMIM, self).__init__(config)
        self.image_size = config.img_size
        self.enable_recompute = config.enable_recompute
        # self.num_classes = num_classes = config.num_classes
        self.embed_dim = embed_dim = config.embed_dim
        self.swiglu = config.swiglu
        self.naiveswiglu = config.naiveswiglu
        # use_mean_pooling = config.use_mean_pooling
        norm_layer = partial(FusedLayerNorm, epsilon=1e-6)
        self.num_heads = num_heads =config.num_heads

        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        init_data = paddle.zeros(shape=[1, 1, embed_dim])
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim],
            default_initializer=paddle.nn.initializer.Assign(init_data))
        self.mask_token = self.create_parameter(
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

        self.stop_grad_conv1 = config.stop_grad_conv1
        # TODO
        if self.stop_grad_conv1:
            self.patch_embed.proj.weight.stop_gradient = True
            self.patch_embed.proj.bias.stop_gradient = True

        if config.use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if config.use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)

        if config.rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = config.img_size // config.patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
                ft_seq_len=None)
        else:
            self.rope = None

        dpr = [
            x.item()
            for x in paddle.linspace(0, config.drop_path_rate, config.depth)
        ]
        self.blocks = paddle.nn.LayerList(sublayers=[
            Block(
                config,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=self.patch_embed.patch_shape if config.use_rel_pos_bias else None,
                rope=self.rope) for i in range(config.depth)
        ])

        self.deepnorm = config.postnorm
        self.norm = norm_layer(embed_dim) if not self.deepnorm else nn.Identity()
        
        self.init_std = config.init_std
        if dist.get_world_size() > 1:
            self.lm_head = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim,
                config.predict_feature_dim,
                weight_attr=None,
                has_bias=True,
                gather_output=True
            )
        else:
            self.lm_head = paddle.nn.Linear(embed_dim, config.predict_feature_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)

        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.lm_head, fleet.meta_parallel.ColumnParallelLinear):
            trunc_normal_(self.lm_head.weight, std=self.init_std)

        if config.xavier_normal_init:
            self.apply(self._xavier_normal_init)
            w = self.patch_embed.proj.weight
            paddle.nn.initializer.XavierNormal(w.reshape([w.shape[0], -1]))
        else:    # ori BEiT init
            self.apply(self._init_weights)
            self.fix_init_weight()

        self.postnorm = config.postnorm
        if self.postnorm:
            self._reinit_respostnorm_ln()

        if self.deepnorm:
            init_scale = math.pow(8.0 * config.depth, 0.25)
            for name, p in self.named_parameters():
                if ('mlp.fc' in name or 'mlp.w' in name or
                        'attn.proj' in name or 'attn.v_proj' in name):
                    print('deepnorm rescale:', name, '/', init_scale)
                    #p.divide(init_scale)
                    p.set_value(p.divide(paddle.to_tensor(init_scale)))

        self.subln = config.subln
        if self.subln and self.naiveswiglu:
            # only B/L
            init_scale = math.sqrt(math.log(config.depth * 2))
            for name, p in self.named_parameters():
                if ('mlp.fc' in name or 'mlp.w' in name or
                        'attn.proj' in name or 'attn.v_proj' in name):
                    print('subln rescale:', name, 'x', init_scale)
                    p.set_value(p.scale(init_scale))

    def _reinit_respostnorm_ln(self):
        zeros_params = paddle.nn.initializer.Constant(0.0)
        for blk in self.blocks:
            zeros_params(blk.norm1.bias)
            zeros_params(blk.norm1.weight)
            zeros_params(blk.norm2.bias)
            zeros_params(blk.norm2.weight)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            #param = param.divide(paddle.to_tensor(math.sqrt(2.0 * layer_id)))
            origin_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float32")
            tmp = paddle.to_tensor(math.sqrt(2.0 * layer_id))
            paddle.set_default_dtype(origin_dtype)
            if origin_dtype != 'float32':
                tmp = tmp.astype(origin_dtype)
            param = param.divide(tmp)

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
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
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                zeros_params(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            zeros_params(m.bias)
            ones_params(m.weight)
        # new added
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                zeros_params(m.bias)

    def _xavier_normal_init(self, m):
        zeros_params = paddle.nn.initializer.Constant(0.0)
        ones_params = paddle.nn.initializer.Constant(1.0)
        if isinstance(m, (paddle.nn.Linear,
                          fleet.meta_parallel.ColumnParallelLinear)):
            paddle.nn.initializer.XavierNormal(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_params(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_params(m.bias)
            ones_params(m.weight)

    def set_grad_checkpointing(self, enable=True):
        self.enable_recompute = enable

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_final_patch_size(self):
        return self.patch_embed.patch_size

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos=False):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        #np.save("pd/patch_embed.npy", x.detach().cpu().numpy())

        # if self.stop_grad_conv1: # do not work when use sharding training
        #     x = x.detach() * 0.9 + x * 0.1

        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand(shape=[batch_size, -1, -1])
        mask_token = self.mask_token.expand(shape=[batch_size, seq_len, -1])

        w = bool_masked_pos.unsqueeze(axis=-1).astype(dtype=mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.pos_drop(x)
        #np.save("pd/pos_drop.npy", x.detach().cpu().numpy())

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
            #np.save("pd/blocks_{}.npy".format(cnt), x.detach().cpu().numpy())

        return self.norm(x)

    def forward(self, x, bool_masked_pos=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        #np.save("pd/forward_features.npy", x.detach().cpu().numpy())
        x = x[:, 1:]
        x = self.lm_head(x[bool_masked_pos])
        #np.save("pd/lm_head.npy", x.detach().cpu().numpy())
        return x


class EVA02ForPretrainConfig(PretrainedConfig):
    model_type = "eva02_for_pretrain"
    is_composition = True

    def __init__(
            self,
            evaclip_config=None,
            eva02_config=None,
            **kwargs, ):
        super().__init__(**kwargs)
        if evaclip_config is None:
            evaclip_config = {}
            logger.info(
                "evaclip_config is None. initializing the EVACLIPConfig with default values."
            )
        if eva02_config is None:
            eva02_config = {}
            logger.info(
                "eva02_config is None. Initializing the EVA02Config with default values."
            )
        self.evaclip_config = evaclip_config
        self.eva02_config = eva02_config

        self.freeze_evaclip = kwargs.get('freeze_evaclip', True)

    @classmethod
    def from_evaclip_eva02_configs(
        cls,
        evaclip_config: EVACLIPConfig,
        eva02_config: EVA02VisionTransformerForMIMConfig,
        **kwargs,
    ):
        return cls(
            evaclip_config=evaclip_config,
            eva02_config=eva02_config,
            **kwargs,
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output["evaclip_config"] = self.evaclip_config
        output["eva02_config"] = self.eva02_config
        return output


class EVA02ForPretrain(PretrainedModel):
    config_class = EVA02ForPretrainConfig

    def __init__(self, config: EVA02ForPretrainConfig):
        super().__init__(config)
        # self.evaclip = EVACLIP.from_pretrained(
        #     pretrained_model_name_or_path=config.evaclip_config)
        # self.evaclip = EVACLIP.from_pretrained("EVA/EVA01-CLIP-g-14/", ignore_mismatched_sizes=True)
        self.evaclip = EVACLIP.from_pretrained(config.evaclip_config, ignore_mismatched_sizes=True)

        self.beit_like = True
        self.freeze_evaclip = config.freeze_evaclip
        if self.freeze_evaclip:
            for name, param in self.evaclip.named_parameters():
                param.stop_gradient = True
            self.evaclip.eval()
            logger.info("freeze evaclip encoder")

        # self.eva02_vit = EVA02VisionTransformerForMIM.from_pretrained(
        #     "EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14/", ignore_mismatched_sizes=True)
        self.eva02_vit = EVA02VisionTransformerForMIM.from_pretrained(
            config.eva02_config, ignore_mismatched_sizes=True)
        # TODO: submit a random init weight to bos?

    def forward(self,
                samples: paddle.Tensor,
                images: paddle.Tensor,
                bool_masked_pos: paddle.Tensor,
                **kwargs):
        if 0:
            samples = paddle.to_tensor(np.load(('samples_bs100.npy'))) # .astype(np.float32))
            images = paddle.to_tensor(np.load(('images_bs100.npy')))
            bool_masked_pos = paddle.to_tensor(np.load(('bool_masked_pos_bs100.npy'))) # .astype(np.int64))

        if self.beit_like:
            with paddle.no_grad(), paddle.amp.auto_cast(enable=False):
                clip_features = self.evaclip.encode_image(images) # [100, 256, 1024], not [100, 1024]
                bool_masked_pos = bool_masked_pos.flatten(start_axis=1).cast('bool') # [100, 256]
                labels = clip_features[bool_masked_pos] # [10458, 1024]

            with paddle.amp.auto_cast(enable=False):
                outputs = self.eva02_vit(samples, bool_masked_pos=bool_masked_pos) # [10458, 1024]

            loss = compute_loss(outputs, labels)

            # print('clip_feature ', clip_features.sum().item())
            # print('bool_masked_pos', bool_masked_pos.sum().item())
            # print('labels ', labels.sum().item())
            # print('outputs', outputs.sum().item())
            # print('loss', loss.sum().item())
            # exit()
        else:
            raise ValueError
        return loss
        

def compute_loss(output, label):
    loss_func = paddle.nn.CosineSimilarity(axis=-1)
    loss = loss_func(
        output.astype(dtype='float32'), label.astype(dtype='float32'))
    return -loss.mean()
