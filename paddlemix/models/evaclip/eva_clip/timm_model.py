import paddle
from . import timm_ext
""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict
from .utils import freeze_batch_norm_2d


class TimmModel(paddle.nn.Layer):
    """ timm model adapter
    # FIXME this adapter is a work in progress, may change in ways that break weight compat
    """

    def __init__(self,
                 model_name,
                 embed_dim,
                 image_size=224,
                 pool='avg',
                 proj='linear',
                 proj_bias=False,
                 drop=0.0,
                 pretrained=False):
        super().__init__()
        self.image_size = timm_ext.to_2tuple(image_size)
        # may never use, just leave it
        timm = None
        self.trunk = timm.create_model(model_name, pretrained=pretrained)
        feat_size = self.trunk.default_cfg.get('pool_size', None)
        feature_ndim = 1 if not feat_size else 2
        if pool in ('abs_attn', 'rot_attn'):
            assert feature_ndim == 2
            self.trunk.reset_classifier(0, global_pool='')
        else:
            reset_kwargs = dict(global_pool=pool) if pool else {}
            self.trunk.reset_classifier(0, **reset_kwargs)
        prev_chs = self.trunk.num_features
        head_layers = OrderedDict()
        if pool == 'abs_attn':
            head_layers['pool'] = timm_ext.AttentionPool2d(
                prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = timm_ext.RotAttentionPool2d(
                prev_chs, out_features=embed_dim)
            prev_chs = embed_dim
        else:
            assert proj, 'projection layer needed if non-attention pooling is used.'
        if proj == 'linear':
            head_layers['drop'] = paddle.nn.Dropout(p=drop)
            head_layers['proj'] = paddle.nn.Linear(
                in_features=prev_chs,
                out_features=embed_dim,
                bias_attr=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = timm_ext.Mlp(prev_chs,
                                              2 * embed_dim,
                                              embed_dim,
                                              drop=drop,
                                              bias=(True, proj_bias))
        self.head = paddle.nn.Sequential(head_layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            for param in self.trunk.parameters():
                param.stop_gradient = not False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            matcher = self.trunk.group_matcher()
            gparams = timm_ext.group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    # may never use
                    self.trunk.get_parameter(param).stop_gradient = not False
            if freeze_bn_stats:
                gmodules = timm_ext.group_modules(
                    self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning(
                'grad checkpointing not supported for this timm image tower, continuing without...'
            )

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x
