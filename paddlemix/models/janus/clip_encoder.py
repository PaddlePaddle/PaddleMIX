# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Optional, Tuple, Union

import paddle

from paddlemix.models.clip.vit_model import VisionTransformer

from .siglip_vit import SigLIPVisionCfg, SigLIPVisionTransformer

SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 336,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
    },
}


def create_siglip_vit(
    model_name: str = "siglip_so400m_patch14_384",
    image_size: int = 384,
    select_layer: int = -1,
    ckpt_path: str = "",
    **kwargs
):
    assert model_name in SigLIP_MODEL_CONFIG.keys(), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"
    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])
    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)
    model = SigLIPVisionTransformer(
        img_size=image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        ignore_head=kwargs.get("ignore_head", True),
        weight_init=kwargs.get("weight_init", "skip"),
        num_classes=0,
    )
    if ckpt_path:
        state_dict = paddle.load(path=str(ckpt_path))
        incompatible_keys = model.set_state_dict(state_dict=state_dict)
        print(f"SigLIP-ViT restores from {ckpt_path},\n\tincompatible_keys:', {incompatible_keys}.")
    return model


class CLIPVisionTower(paddle.nn.Layer):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers
        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(vision_tower_params)
        if pixel_mean is not None and pixel_std is not None:
            image_norm = paddle.vision.transforms.Normalize(mean=pixel_mean, std=pixel_std)
        else:
            image_norm = None
        self.image_norm = image_norm

    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):
            self.select_feature = "same"
            vision_tower = create_siglip_vit(**vision_tower_params)
            forward_kwargs = dict()
        else:
            vision_tower = VisionTransformer.from_pretrained(**vision_tower_params)
            forward_kwargs = dict(output_hidden_states=True)
        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, paddle.Tensor):
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "same":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        """

        Args:
            images (torch.Tensor): [b, 3, H, W]

        Returns:
            image_features (torch.Tensor): [b, n_patch, d]
        """
        if self.image_norm is not None:
            images = self.image_norm(images)
        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)
        image_features = self.feature_select(image_forward_outs)
        return image_features
