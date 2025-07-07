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

import numpy as np
import paddle
import paddle.nn as nn

from ..audiomae import mae as models_mae


class Vanilla_AudioMAE(nn.Layer):
    """Audio Masked Autoencoder (MAE) pre-trained on AudioSet (for AudioLDM2)"""

    def __init__(
        self,
    ):
        super().__init__()
        model = models_mae.__dict__["mae_vit_base_patch16"](in_chans=1, audio_exp=True, img_size=(1024, 128))

        self.model = model.eval()

    def forward(self, x, mask_ratio=0.0, no_mask=False, no_average=False):
        """
        x: mel fbank [Batch, 1, 1024 (T), 128 (F)]
        mask_ratio: 'masking ratio (percentage of removed patches).'
        """
        with paddle.no_grad():
            # embed: [B, 513, 768] for mask_ratio=0.0
            if no_mask:
                if no_average:
                    raise RuntimeError("This function is deprecated")
                else:
                    embed = self.model.forward_encoder_no_mask(x)  # mask_ratio
            else:
                raise RuntimeError("This function is deprecated")
        return embed


class AudioMAEConditionCTPoolRand(nn.Layer):
    def __init__(
        self,
        time_pooling_factors=[1, 2, 4, 8],
        freq_pooling_factors=[1, 2, 4, 8],
        eval_time_pooling=None,
        eval_freq_pooling=None,
        mask_ratio=0.0,
        regularization=False,
        no_audiomae_mask=True,
        no_audiomae_average=False,
    ):
        super().__init__()
        self.device = None
        self.time_pooling_factors = time_pooling_factors
        self.freq_pooling_factors = freq_pooling_factors
        self.no_audiomae_mask = no_audiomae_mask
        self.no_audiomae_average = no_audiomae_average

        self.eval_freq_pooling = eval_freq_pooling
        self.eval_time_pooling = eval_time_pooling
        self.mask_ratio = mask_ratio
        self.use_reg = regularization

        self.audiomae = Vanilla_AudioMAE()
        self.audiomae.eval()
        for p in self.audiomae.parameters():
            p.stop_gradient = True

    # Required
    def get_unconditional_condition(self, batchsize):
        param = self.audiomae.parameters()[0]
        assert param.stop_gradient

        time_pool, freq_pool = min(self.eval_time_pooling, 64), min(self.eval_freq_pooling, 8)

        token_num = int(512 / (time_pool * freq_pool))
        return [
            paddle.zeros((batchsize, token_num, 768), dtype="float32"),
            paddle.ones((batchsize, token_num), dtype="float32"),
        ]

    def pool(self, representation, time_pool=None, freq_pool=None):
        assert representation.shape[-1] == 768
        representation = representation[:, 1:, :]
        perm = list(range(representation.dim()))
        new_perm = perm
        new_perm[1], new_perm[2] = perm[2], perm[1]
        representation = representation.transpose(new_perm)
        bs, embedding_dim, token_num = representation.shape
        representation = representation.reshape([bs, embedding_dim, 64, 8])

        if self.training:
            if time_pool is None and freq_pool is None:
                time_pool = min(
                    64,
                    self.time_pooling_factors[np.random.choice(list(range(len(self.time_pooling_factors))))],
                )
                freq_pool = min(8, time_pool)  # TODO here I make some modification.
        else:
            time_pool, freq_pool = min(self.eval_time_pooling, 64), min(self.eval_freq_pooling, 8)

        self.avgpooling = nn.AvgPool2D(kernel_size=(time_pool, freq_pool), stride=(time_pool, freq_pool))
        self.maxpooling = nn.MaxPool2D(kernel_size=(time_pool, freq_pool), stride=(time_pool, freq_pool))

        pooled = (
            self.avgpooling(representation) + self.maxpooling(representation)
        ) / 2  # [bs, embedding_dim, time_token_num, freq_token_num]
        pooled = pooled.flatten(2).transpose([0, 2, 1])
        return pooled  # [bs, token_num, embedding_dim]

    def regularization(self, x):
        assert x.shape[-1] == 768
        x = nn.functional.normalize(x, p=2, axis=-1)
        return x

    # Required
    def forward(self, batch, time_pool=None, freq_pool=None):
        assert batch.shape[-2] == 1024 and batch.shape[-1] == 128

        batch = batch.unsqueeze(1)
        with paddle.no_grad():
            representation = self.audiomae(
                batch,
                mask_ratio=self.mask_ratio,
                no_mask=self.no_audiomae_mask,
                no_average=self.no_audiomae_average,
            )

            representation = self.pool(representation, time_pool, freq_pool)
            if self.use_reg:
                representation = self.regularization(representation)
            return [
                representation,
                paddle.ones((representation.shape[0], representation.shape[1]), dtype="float32"),
            ]
