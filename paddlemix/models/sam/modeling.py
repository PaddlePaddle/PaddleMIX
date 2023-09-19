# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Dict, List

import numpy as np
import paddle
from paddlenlp.transformers.model_utils import register_base_model

from paddlemix.models.model_utils import MixPretrainedModel

from .configuration import SamConfig
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

__all__ = [
    "SamModel",
    "SamPretrainedModel",
]


class SamPretrainedModel(MixPretrainedModel):
    """
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = SamConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "Sam"


@register_base_model
class SamModel(SamPretrainedModel):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, config: SamConfig):
        super(SamModel, self).__init__(config)

        prompt_embed_dim = config.prompt_embed_dim
        image_size = config.image_size
        vit_patch_size = config.vit_patch_size
        image_embedding_size = image_size // vit_patch_size
        assert config.input_type is not None, "input_type is None, but it is required."
        self.input_type = config.input_type
        self.set_image = False
        self.image_encoder = ImageEncoderViT(
            depth=config.encoder_depth,
            embed_dim=config.encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(paddle.nn.LayerNorm, epsilon=1e-6),
            num_heads=config.encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=config.encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.eval()
        self.register_buffer(
            "pixel_mean",
            paddle.to_tensor(config.pixel_mean).reshape([-1, 1, 1]),
            persistable=False,
        )
        self.register_buffer(
            "pixel_std",
            paddle.to_tensor(config.pixel_std).reshape([-1, 1, 1]),
            persistable=False,
        )

    @property
    def device(self) -> Any:
        if paddle.is_compiled_with_cuda():
            return "gpu"
        else:
            return "cpu"

    def reset_img(self):
        self.features = None
        self.set_image = False

    def after_forward(self):
        # masks = masks[0].detach().cpu().numpy()
        # iou_predictions = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks = low_res_masks[0].detach().cpu().numpy()
        pass

    @paddle.no_grad()
    def prompt_forward_point(self, x=None, coords_paddle=None):
        labels_paddle = np.array([1])
        labels_paddle = paddle.to_tensor(labels_paddle).cast("int32")
        labels_paddle = labels_paddle[None, :]
        points = (coords_paddle, labels_paddle)

        if self.set_image is False or x is not None:
            self.reset_img()
            self.features = self.image_encoder(x)  # [1, 3, 1024, 1024]
            self.set_image = True

        # Embed prompts

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks

    @paddle.no_grad()
    def prompt_forward_box(self, x=None, box_paddle=None):
        if self.set_image is False or x is not None:
            self.reset_img()
            self.features = self.image_encoder(x)
            self.set_image = True

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_paddle,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks  # , iou_predictions, low_res_masks

    @paddle.no_grad()
    def full_mask_forward(self, img: List[Dict[str, Any]], coords_paddle):
        labels_paddle = paddle.ones(
            shape=[
                coords_paddle.shape[0],
            ],
            dtype="int64",
        )
        labels_paddle = paddle.to_tensor(labels_paddle).cast("int32")[:, None]

        points = (coords_paddle, labels_paddle)
        if self.set_image is False:
            self.features = self.image_encoder(img)
            self.set_image = True

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks, iou_predictions  # (64, 3) # low_res_masks,

    def forward(self, img=None, prompt=None):
        if self.input_type == "points":
            masks = self.prompt_forward_point(x=img, coords_paddle=prompt)
        elif self.input_type == "boxs":
            masks = self.prompt_forward_box(x=img, box_paddle=prompt)
        elif self.input_type == "points_grid":
            masks, iou_predictions = self.full_mask_forward(img, prompt)
            return masks, iou_predictions
        else:
            NotImplementedError(
                'input_type need to be in ["points", "boxs", "points_grid"], but got: {}'.format(self.input_type)
            )

        return masks
