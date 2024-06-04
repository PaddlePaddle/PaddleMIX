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

import paddle

from ppdiffusers.transformers import CLIPVisionModel

from .resampler import Resampler

BATCH_SIZE = 1
OUTPUT_DIM = 1280
NUM_QUERIES = 8
NUM_LATENTS_MEAN_POOLED = 4  # 0 for no mean pooling (previous behavior)
APPLY_POS_EMB = True  # False for no positional embeddings (previous behavior)
IMAGE_ENCODER_NAME_OR_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def main():
    image_encoder = CLIPVisionModel.from_pretrained(IMAGE_ENCODER_NAME_OR_PATH)
    embedding_dim = image_encoder.config.hidden_size
    print(f"image_encoder hidden size: {embedding_dim}")

    image_proj_model = Resampler(
        dim=1024,
        depth=2,
        dim_head=64,
        heads=16,
        num_queries=NUM_QUERIES,
        embedding_dim=embedding_dim,
        output_dim=OUTPUT_DIM,
        ff_mult=2,
        max_seq_len=257,
        apply_pos_emb=APPLY_POS_EMB,
        num_latents_mean_pooled=NUM_LATENTS_MEAN_POOLED,
    )

    dummy_images = paddle.randn([BATCH_SIZE, 3, 224, 224])
    with paddle.no_grad():
        image_embeds = image_encoder(dummy_images, output_hidden_states=True).hidden_states[-2]
    print("image_embds shape: ", image_embeds.shape)

    with paddle.no_grad():
        ip_tokens = image_proj_model(image_embeds)
    print("ip_tokens shape:", ip_tokens.shape)
    assert ip_tokens.shape == (BATCH_SIZE, NUM_QUERIES + NUM_LATENTS_MEAN_POOLED, OUTPUT_DIM)


if __name__ == "__main__":
    main()
