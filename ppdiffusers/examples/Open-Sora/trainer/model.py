# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
from models.stdit.stdit2 import STDiT2
from models.text_encoder import T5Encoder
from models.vae import VideoAutoencoderKL
from paddlenlp.utils.log import logger
from schedulers.iddpm import IDDPM
from utils.train_utils import MaskGenerator

from ppdiffusers.training_utils import freeze_params


class OpenSoraModel(nn.Layer):
    def __init__(self, model_args, data_args):
        super().__init__()

        self.text_encoder = T5Encoder(
            from_pretrained=model_args.text_encoder_path, model_max_length=model_args.text_encoder_model_max_length
        )

        self.vae = VideoAutoencoderKL(
            from_pretrained=model_args.vae_model_path, micro_batch_size=model_args.vae_micro_batch_size
        )

        input_size = (data_args.num_frames, data_args.train_height, data_args.train_width)
        latent_size = self.vae.get_latent_size(input_size)

        self.model = STDiT2.from_pretrained(
            pretrained_model_name_or_path=model_args.stdit2_pretrained_path,
            input_size=latent_size,
            in_channels=self.vae.out_channels,
            caption_channels=self.text_encoder.output_dim,
            model_max_length=self.text_encoder.model_max_length,
            dtype="float32",
        )

        self.scheduler = IDDPM(
            timestep_respacing=model_args.timestep_respacing,
        )

        self.mask_ratios = {
            "mask_no": 0.75,
            "mask_quarter_random": 0.025,
            "mask_quarter_head": 0.025,
            "mask_quarter_tail": 0.025,
            "mask_quarter_head_tail": 0.05,
            "mask_image_random": 0.025,
            "mask_image_head": 0.025,
            "mask_image_tail": 0.025,
            "mask_image_head_tail": 0.05,
        }

        if self.mask_ratios is not None:
            self.mask_generator = MaskGenerator(self.mask_ratios)

        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.t5.model.parameters())
        logger.info("Freeze vae n text_encoder parameters")
        self.vae.eval()
        self.text_encoder.t5.model.eval()
        self.model.train()

    def forward(self, batch, **kwargs):
        x = batch.pop("video")  # [B, C, T, H, W]
        y = batch.pop("text")

        # Visual and text encoding
        with paddle.no_grad():
            # Prepare visual inputs
            x = self.vae.encode(x)  # [B, C, T, H/P, W/P]
            # Prepare text inputs
            model_args = self.text_encoder.encode(y)

        # Mask
        if self.mask_ratios is not None:
            mask = self.mask_generator.get_masks(x)
            model_args["x_mask"] = mask
        else:
            mask = None

        # Video info
        for k, v in batch.items():
            model_args[k] = v

        # Diffusion
        t = paddle.randint(
            0,
            self.scheduler.num_timesteps,
            [
                x.shape[0],
            ],
        )
        loss_dict = self.scheduler.training_losses(self.model, x, t, model_args, mask=mask)

        # Backward & update
        loss = loss_dict["loss"].mean()

        return loss
