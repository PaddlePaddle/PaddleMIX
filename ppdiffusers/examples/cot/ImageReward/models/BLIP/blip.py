# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
 * Adapted from BLIP (https://github.com/salesforce/BLIP)
"""

import warnings

from paddlenlp.transformers import BertTokenizer

warnings.filterwarnings("ignore")

from .vit import VisionTransformer


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ["base", "large"], "vit parameter must be base or large"
    if vit == "base":
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate,
        )
    elif vit == "large":
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate,
        )
    return visual_encoder, vision_width
