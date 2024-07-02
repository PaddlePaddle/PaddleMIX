# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from functools import partial
from typing import Dict, Union

import paddle
import paddle.nn as nn

from ..clip.eva_clip_model import EVACLIP, EVACLIPConfig
from ..clip.modules.rope import VisionRotaryEmbeddingFast
from ..clip.utils import trunc_normal_
from .modeling_finetune import (
    Block,
    DecoupledRelativePositionBias,
    PatchEmbed,
    RelativePositionBias,
)

try:
    from ..clip.modules.fusedln import FusedLayerNorm
except:
    from paddle.nn import LayerNorm as FusedLayerNorm

    print("Warning, FusedLn module is not available, use LayerNorm instead.")

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

from paddlemix.models.model_utils import MixPretrainedModel

__all__ = [
    "EVA02VisionTransformerForMIMPretrainedModel",
    "EVA02VisionTransformerForMIM",
    "EVA02ForPretrain",
]


class EVA02VisionTransformerForMIMConfig(PretrainedConfig):

    model_type = "eva02_vit_pretrain"
    attribute_map: Dict[str, str] = {}

    def __init__(
        self,
        image_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_shared_decoupled_rel_pos_bias=False,
        init_scale=0.001,
        enable_recompute=False,
        stop_grad_conv1=True,  #
        postnorm=False,
        deepnorm=False,  #
        subln=True,
        xattn=False,
        swiglu=False,
        naiveswiglu=False,
        rope=True,
        init_std=0.02,  #
        xavier_normal_init=True,  #
        attn_head_dim=None,  #
        predict_feature_dim=1024,  # default EVA-CLIP-01-g-14, 1024 EVA02-CLIP-bigE-14, 768 EVA02-CLIP-L-14
        fusedLN=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.layers = layers
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
        self.init_scale = init_scale
        self.enable_recompute = enable_recompute
        self.stop_grad_conv1 = stop_grad_conv1
        self.deepnorm = deepnorm
        self.postnorm = postnorm
        self.xattn = xattn
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu
        self.rope = rope
        self.subln = subln
        self.fusedLN = fusedLN
        self.init_std = init_std
        self.xavier_normal_init = xavier_normal_init
        self.attn_head_dim = attn_head_dim
        self.predict_feature_dim = predict_feature_dim

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class EVA02VisionTransformerForMIMPretrainedModel(MixPretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.

    """

    model_config_file = "config.json"
    config_class = EVA02VisionTransformerForMIMConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "eva02_vit_pretrain"


class EVA02VisionTransformerForMIM(EVA02VisionTransformerForMIMPretrainedModel):
    def __init__(self, config: EVA02VisionTransformerForMIMConfig):
        super(EVA02VisionTransformerForMIM, self).__init__(config)
        self.image_size = config.image_size
        self.enable_recompute = config.enable_recompute
        self.embed_dim = embed_dim = config.embed_dim
        self.swiglu = config.swiglu
        self.naiveswiglu = config.naiveswiglu
        norm_layer = partial(FusedLayerNorm, epsilon=1e-6)
        self.num_heads = num_heads = config.num_heads

        self.predict_feature_dim = config.predict_feature_dim
        # will be reset by teacher evaclip's text.output_dim before training

        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        init_data = paddle.zeros(shape=[1, 1, embed_dim])
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=paddle.nn.initializer.Assign(init_data)
        )
        self.mask_token = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=paddle.nn.initializer.Assign(init_data)
        )
        if config.use_abs_pos_emb:
            init_data = paddle.zeros(shape=[1, num_patches + 1, embed_dim])
            self.pos_embed = self.create_parameter(
                shape=[1, num_patches + 1, embed_dim], default_initializer=paddle.nn.initializer.Assign(init_data)
            )
        else:
            self.pos_embed = None
        self.pos_drop = paddle.nn.Dropout(p=config.drop_rate)

        # TODO
        self.stop_grad_conv1 = config.stop_grad_conv1
        if self.stop_grad_conv1:
            self.patch_embed.proj.weight.stop_gradient = True
            self.patch_embed.proj.bias.stop_gradient = True

        if config.use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if config.use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )

        if config.rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = config.image_size // config.patch_size
            self.rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, ft_seq_len=None)
        else:
            self.rope = None

        dpr = [x.item() for x in paddle.linspace(0, config.drop_path_rate, config.layers)]
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    config,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    window_size=self.patch_embed.patch_shape if config.use_rel_pos_bias else None,
                    rope=self.rope,
                )
                for i in range(config.layers)
            ]
        )

        self.deepnorm = config.postnorm
        self.norm = norm_layer(embed_dim) if not self.deepnorm else nn.Identity()

        self.init_std = config.init_std
        if dist.get_world_size() > 1:
            self.lm_head = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim, self.predict_feature_dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.lm_head = paddle.nn.Linear(embed_dim, self.predict_feature_dim)

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
        else:  # ori BEiT init
            self.apply(self._init_weights)
            self.fix_init_weight()

        self.postnorm = config.postnorm
        if self.postnorm:
            self._reinit_respostnorm_ln()

        if self.deepnorm:
            init_scale = math.pow(8.0 * config.layers, 0.25)
            for name, p in self.named_parameters():
                if "mlp.fc" in name or "mlp.w" in name or "attn.proj" in name or "attn.v_proj" in name:
                    print("deepnorm rescale:", name, "/", init_scale)
                    p.set_value(p.divide(paddle.to_tensor(init_scale)))

        self.subln = config.subln
        if self.subln and self.naiveswiglu:
            # only B/L
            init_scale = math.sqrt(math.log(config.layers * 2))
            for name, p in self.named_parameters():
                if "mlp.fc" in name or "mlp.w" in name or "attn.proj" in name or "attn.v_proj" in name:
                    print("subln rescale:", name, "x", init_scale)
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
            origin_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float32")
            tmp = paddle.to_tensor(math.sqrt(2.0 * layer_id))
            paddle.set_default_dtype(origin_dtype)
            if origin_dtype != "float32":
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
        if isinstance(m, (paddle.nn.Linear, fleet.meta_parallel.ColumnParallelLinear)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                zeros_params(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            zeros_params(m.bias)
            ones_params(m.weight)
        # pretrain new added
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                zeros_params(m.bias)

    def _xavier_normal_init(self, m):
        zeros_params = paddle.nn.initializer.Constant(0.0)
        ones_params = paddle.nn.initializer.Constant(1.0)
        if isinstance(m, (paddle.nn.Linear, fleet.meta_parallel.ColumnParallelLinear)):
            paddle.nn.initializer.XavierNormal(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_params(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_params(m.bias)
            ones_params(m.weight)

    def set_grad_checkpointing(self, enable=True):
        self.enable_recompute = enable

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_final_patch_size(self):
        return self.patch_embed.patch_size

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos=False):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)

        # if self.stop_grad_conv1: # TODO
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

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        cnt = 0
        for blk in self.blocks:
            cnt += 1
            if self.enable_recompute:
                x = paddle.distributed.fleet.utils.recompute(blk, x, rel_pos_bias, use_reentrant=False)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, image, bool_masked_pos=False):
        x = self.forward_features(image, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        x = self.lm_head(x[bool_masked_pos])
        return x


class EVA02ForPretrainConfig(PretrainedConfig):
    model_type = "eva02_for_pretrain"

    def __init__(
        self,
        teacher_config={},
        student_config={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.teacher_config = teacher_config
        self.student_config = student_config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike] = None,
        pretrained_teacher_name_or_path: Union[str, os.PathLike] = None,
        pretrained_student_name_or_path: Union[str, os.PathLike] = None,
        **kwargs,
    ) -> "PretrainedConfig":
        assert pretrained_model_name_or_path is not None or (
            pretrained_teacher_name_or_path is not None and pretrained_student_name_or_path is not None
        ), (
            f"Either `pretrained_model_name_or_path` or (`pretrained_teacher_name_or_path` and `pretrained_student_name_or_path`) must be set, but"
            f"received `pretrained_model_name_or_path={pretrained_model_name_or_path}` and `pretrained_teacher_name_or_path={pretrained_teacher_name_or_path}`, "
            f"`pretrained_student_name_or_path={pretrained_student_name_or_path}`"
        )
        config_dict = {}
        if pretrained_model_name_or_path is not None:
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
            if (
                "model_type" in config_dict
                and hasattr(cls, "model_type")
                and config_dict["model_type"] != cls.model_type
            ):
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )

        if pretrained_teacher_name_or_path is not None:
            teacher_config_dict, kwargs = cls.get_config_dict(pretrained_teacher_name_or_path, **kwargs)
            if "model_type" in teacher_config_dict and teacher_config_dict["model_type"] != "evaclip":
                logger.warning(
                    f"You are using a model of type {teacher_config_dict['model_type']} to instantiate a model of type "
                    f"evaclip. This is not supported for all configurations of models and can yield errors."
                )
            config_dict["teacher_config"] = teacher_config_dict

        if pretrained_student_name_or_path is not None:
            student_config_dict, kwargs = cls.get_config_dict(pretrained_student_name_or_path, **kwargs)
            if "model_type" in student_config_dict and student_config_dict["model_type"] != "eva02_vit_pretrain":
                logger.warning(
                    f"You are using a model of type {student_config_dict['model_type']} to instantiate a model of type "
                    f"eva02_vit_pretrain. This is not supported for all configurations of models and can yield errors."
                )
            config_dict["student_config"] = student_config_dict

        return cls.from_dict(config_dict, **kwargs)


class EVA02ForPretrainModel(MixPretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVA02ForPretrainConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "eva02_for_pretrain"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        pretrained_teacher_name_or_path=None,
        pretrained_student_name_or_path=None,
        from_hf_hub: bool = False,
        subfolder: str = "",
        *args,
        **kwargs,
    ):
        assert pretrained_model_name_or_path is not None or (
            pretrained_teacher_name_or_path is not None and pretrained_student_name_or_path is not None
        ), (
            f"Either `pretrained_model_name_or_path` or (`pretrained_teacher_name_or_path` and `pretrained_student_name_or_path`) must be set, but"
            f"received `pretrained_model_name_or_path={pretrained_model_name_or_path}` and `pretrained_teacher_name_or_path={pretrained_teacher_name_or_path}`, "
            f"`pretrained_student_name_or_path={pretrained_student_name_or_path}`"
        )

        if pretrained_model_name_or_path is not None:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                from_hf_hub=from_hf_hub,
                subfolder=subfolder,
                *args,
                **kwargs,
            )
        else:
            config_dict = {
                "teacher_config": pretrained_teacher_name_or_path,
                "student_config": pretrained_student_name_or_path,
            }
            config = EVA02ForPretrainConfig.from_dict(config_dict)
            return cls(config, *args, **kwargs)


class EVA02ForPretrain(EVA02ForPretrainModel):
    def __init__(self, config, beit_like=True, freeze_teacher=True, enable_recompute=True):
        super().__init__(config)
        self.beit_like = beit_like
        self.freeze_teacher = freeze_teacher

        if isinstance(config.teacher_config, str):
            self.teacher = EVACLIP.from_pretrained(
                config.teacher_config, ignore_mismatched_sizes=True
            )  # for MP loading
        else:
            teacher_config = EVACLIPConfig(**config.teacher_config)
            self.teacher = EVACLIP(teacher_config)

        # must be set
        self.teacher.visual.output_tokens = True
        self.teacher.visual.token_feats = True

        if isinstance(config.student_config, str):
            self.student = EVA02VisionTransformerForMIM.from_pretrained(
                config.student_config, ignore_mismatched_sizes=True
            )
        else:
            student_config = EVA02VisionTransformerForMIMConfig(**config.student_config)
            student_config.predict_feature_dim = self.teacher.text.output_dim
            self.student = EVA02VisionTransformerForMIM(student_config)

        if self.freeze_teacher:
            for name, param in self.teacher.named_parameters():
                param.stop_gradient = True
            self.teacher.eval()
            logger.info("freeze teacher evaclip encoder")

        if enable_recompute:
            self.teacher.set_grad_checkpointing(True)
            self.student.set_grad_checkpointing(True)

    def set_grad_checkpointing(self, enable=True):
        self.teacher.set_grad_checkpointing(enable)
        self.student.set_grad_checkpointing(enable)

    def forward(self, samples, image, bool_masked_pos, get_feats=False):
        # [bs, 3, 224, 224] [bs, 3, 224, 224] [bs, 256]
        if self.beit_like:
            with paddle.no_grad(), paddle.amp.auto_cast():
                clip_features = self.teacher.encode_image(image)  # [bs, 256, 1024]
                bool_masked_pos = bool_masked_pos.flatten(start_axis=1).cast("bool")  # [bs, 256]
                labels = clip_features[bool_masked_pos]  # [N, 1024]

            with paddle.amp.auto_cast():
                outputs = self.student(samples, bool_masked_pos=bool_masked_pos)  # [N, 1024]

            loss = compute_loss(outputs, labels)
        else:
            raise ValueError
        if get_feats:
            return outputs
        return loss


def compute_loss(output, label):
    loss_func = paddle.nn.CosineSimilarity(axis=-1)
    loss = loss_func(output.astype(dtype="float32"), label.astype(dtype="float32"))
    return -loss.mean()
