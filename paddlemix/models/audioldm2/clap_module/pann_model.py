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

import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp/"

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.audio.features import Spectrogram

from .feature_fusion import iAFF, AFF, DAF
from .utils import interpolate, LogmelFilterBank, SpecAugmentation

class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            weight_attr=nn.initializer.XavierUniform(),
            bias_attr=False,
        )

        self.conv2 = nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            weight_attr=nn.initializer.XavierUniform(),
            bias_attr=False,
        )

        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x

class Cnn14(nn.Layer):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        enable_fusion=False,
        fusion_type="None",
    ):
        super(Cnn14, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            # freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2D(64)

        if (self.enable_fusion) and (self.fusion_type == "channel_map"):
            self.conv_block1 = ConvBlock(in_channels=4, out_channels=64)
        else:
            self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, weight_attr=nn.initializer.XavierUniform())
        self.fc_audioset = nn.Linear(2048, classes_num, weight_attr=nn.initializer.XavierUniform())

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]
        ):
            self.mel_conv1d = nn.Sequential(
                nn.Conv1D(64, 64, kernel_size=5, stride=3, padding=2),
                nn.BatchNorm1D(64),  # No Relu
            )
            if self.fusion_type == "daf_1d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_1d":
                self.fusion_model = AFF(channels=64, type="1D")
            elif self.fusion_type == "iaff_1d":
                self.fusion_model = iAFF(channels=64, type="1D")

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            self.mel_conv2d = nn.Sequential(
                nn.Conv2D(1, 64, kernel_size=(5, 5), stride=(6, 2), padding=(2, 2)),
                nn.BatchNorm2D(64),
                nn.ReLU(),
            )

            if self.fusion_type == "daf_2d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_2d":
                self.fusion_model = AFF(channels=64, type="2D")
            elif self.fusion_type == "iaff_2d":
                self.fusion_model = iAFF(channels=64, type="2D")

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        if self.enable_fusion and input["longer"].sum() == 0:
            # if no audio is longer than 10s, then randomly select one audio to be longer
            input["longer"][paddle.randint(0, input["longer"].shape[0], (1,))] = True

        if not self.enable_fusion:
            x = self.spectrogram_extractor(
                input["waveform"]
            )  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

            x = x.transpose([0, 3, 2, 1])
            x = self.bn0(x)
            x = x.transpose([0, 3, 2, 1])
        else:
            longer_list = input["longer"]
            x = input["mel_fusion"]
            longer_list_idx = paddle.where(longer_list)[0].squeeze()
            x = x.transpose([0, 3, 2, 1])
            x = self.bn0(x)
            x = x.transpose([0, 3, 2, 1])
            if self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]:
                new_x = x[:, 0:1, :, :].clone()
                # local processing
                if len(longer_list_idx) > 0:
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone()
                    FB, FC, FT, FF = fusion_x_local.shape
                    fusion_x_local = fusion_x_local.reshape([FB * FC, FT, FF])
                    fusion_x_local = paddle.transpose(
                        fusion_x_local, [0, 2, 1]
                    )
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.reshape(
                        [FB, FC, FF, fusion_x_local.shape[-1]]
                    )
                    fusion_x_local = (
                        paddle.transpose(fusion_x_local, [0, 2, 1, 3])
                        .flatten(2)
                    )
                    if fusion_x_local.shape[-1] < FT:
                        fusion_x_local = paddle.concat(
                            [
                                fusion_x_local,
                                paddle.zeros(
                                    (FB, FF, FT - fusion_x_local.shape[-1])
                                ),
                            ],
                            axis=-1,
                        )
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    # 1D fusion
                    new_x = new_x.squeeze(1).transpose([0, 2, 1])
                    new_x[longer_list_idx] = self.fusion_model(
                        new_x[longer_list_idx], fusion_x_local
                    )
                    x = new_x.transpose([0, 2, 1])[:, None, :, :]
                else:
                    x = new_x
            elif self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d", "channel_map"]:
                x = x  # no change

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            global_x = x[:, 0:1, :, :]

            # global processing
            B, C, H, W = global_x.shape
            global_x = self.conv_block1(global_x, pool_size=(2, 2), pool_type="avg")
            if len(longer_list_idx) > 0:
                local_x = x[longer_list_idx, 1:, :, :]
                TH = global_x.shape[-2]
                # local processing
                B, C, H, W = local_x.shape
                local_x = local_x.reshape([B * C, 1, H, W])
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.reshape(
                    [B, C, local_x.shape[1], local_x.shape[2], local_x.shape[3]]
                )
                local_x = local_x.transpose([0, 2, 1, 3, 4]).flatten(2, 3)
                TB, TC, _, TW = local_x.shape
                if local_x.shape[-2] < TH:
                    local_x = paddle.concat(
                        [
                            local_x,
                            paddle.zeros(
                                [TB, TC, TH - local_x.shape[-2], TW]
                            ),
                        ],
                        axis=-2,
                    )
                else:
                    local_x = local_x[:, :, :TH, :]

                global_x[longer_list_idx] = self.fusion_model(
                    global_x[longer_list_idx], local_x
                )
            x = global_x
        else:
            x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = paddle.mean(x, axis=3)

        latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x = latent_x1 + latent_x2
        perm_shape = list(range(latent_x.dim()))
        new_perm_shape = perm_shape
        new_perm_shape[2], new_perm_shape[1] = perm_shape.shape[1], perm_shape.shape[2]
        latent_x = latent_x.transpose(new_perm_shape)
        latent_x = F.relu_(self.fc1(latent_x))
        latent_output = interpolate(latent_x, 32)

        x1 = paddle.max(x, axis=2)
        x2 = paddle.mean(x, axis=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = F.sigmoid(self.fc_audioset(x))

        output_dict = {
            "clipwise_output": clipwise_output,
            "embedding": embedding,
            "fine_grained_embedding": latent_output,
        }
        return output_dict


def create_pann_model(audio_cfg, enable_fusion=False, fusion_type="None"):
    try:
        ModelProto = eval(audio_cfg.model_name)
        model = ModelProto(
            sample_rate=audio_cfg.sample_rate,
            window_size=audio_cfg.window_size,
            hop_size=audio_cfg.hop_size,
            mel_bins=audio_cfg.mel_bins,
            fmin=audio_cfg.fmin,
            fmax=audio_cfg.fmax,
            classes_num=audio_cfg.class_num,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type,
        )
        return model
    except:
        raise RuntimeError(
            f"Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough."
        )
