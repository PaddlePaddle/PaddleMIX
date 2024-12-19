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

import sys

import paddle

from paddlemix.models.diffsinger.utils import paddle_aux
from paddlemix.models.diffsinger.modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
)
from paddlemix.models.diffsinger.modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
)
from paddlemix.models.diffsinger.modules.fastspeech.tts_modules import (
    DurationPredictor,
    FastSpeech2Encoder,
)
from paddlemix.models.diffsinger.utils.hparams import hparams
from paddlemix.models.diffsinger.utils.text_encoder import PAD_INDEX


class FastSpeech2Variance(paddle.nn.Layer):
    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams["predict_dur"]
        self.linguistic_mode = "word" if hparams["predict_dur"] else "phoneme"
        self.txt_embed = Embedding(vocab_size, hparams["hidden_size"], PAD_INDEX)
        if self.predict_dur:
            self.onset_embed = Embedding(2, hparams["hidden_size"])
            self.word_dur_embed = Linear(1, hparams["hidden_size"])
        else:
            self.ph_dur_embed = Linear(1, hparams["hidden_size"])
        self.encoder = FastSpeech2Encoder(
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["enc_layers"],
            ffn_kernel_size=hparams["enc_ffn_kernel_size"],
            ffn_act=hparams["ffn_act"],
            dropout=hparams["dropout"],
            num_heads=hparams["num_heads"],
            use_pos_embed=hparams["use_pos_embed"],
            rel_pos=hparams["rel_pos"],
        )
        dur_hparams = hparams["dur_prediction_args"]
        if self.predict_dur:
            self.midi_embed = Embedding(128, hparams["hidden_size"])
            self.dur_predictor = DurationPredictor(
                in_dims=hparams["hidden_size"],
                n_chans=dur_hparams["hidden_size"],
                n_layers=dur_hparams["num_layers"],
                dropout_rate=dur_hparams["dropout"],
                kernel_size=dur_hparams["kernel_size"],
                offset=dur_hparams["log_offset"],
                dur_loss_type=dur_hparams["loss_type"],
            )

    def forward(self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, spk_embed=None, infer=True):
        """
        :param txt_tokens: (train, infer) [B, T_ph]
        :param midi: (train, infer) [B, T_ph]
        :param ph2word: (train, infer) [B, T_ph]
        :param ph_dur: (train, [infer]) [B, T_ph]
        :param word_dur: (infer) [B, T_w]
        :param spk_embed: (train) [B, T_ph, H]
        :param infer: whether inference
        :return: encoder_out, ph_dur_pred
        """
        txt_embed = self.txt_embed(txt_tokens)
        if self.linguistic_mode == "word":
            b = tuple(txt_tokens.shape)[0]
            onset = paddle.diff(x=ph2word, axis=1, prepend=paddle.zeros(shape=[b, 1], dtype=ph2word.dtype)) > 0
            onset_embed = self.onset_embed(onset.astype(dtype="int64"))
            if word_dur is None or not infer:
                word_dur = paddle.zeros(shape=[b, ph2word.max() + 1], dtype=ph_dur.dtype).put_along_axis(
                    axis=1, indices=ph2word, values=ph_dur, reduce="add"
                )[:, 1:]
            word_dur = paddle.take_along_axis(
                arr=paddle.nn.functional.pad(x=word_dur, pad=[1, 0], value=0, pad_from_left_axis=False),
                axis=1,
                indices=ph2word,
                broadcast=False,
            )
            word_dur_embed = self.word_dur_embed(word_dur.astype(dtype="float32")[:, :, None])
            encoder_out = self.encoder(txt_embed, onset_embed + word_dur_embed, txt_tokens == 0)
        else:
            ph_dur_embed = self.ph_dur_embed(ph_dur.astype(dtype="float32")[:, :, None])
            encoder_out = self.encoder(txt_embed, ph_dur_embed, txt_tokens == 0)
        if self.predict_dur:
            midi_embed = self.midi_embed(midi)
            dur_cond = encoder_out + midi_embed
            if spk_embed is not None:
                dur_cond += spk_embed
            ph_dur_pred = self.dur_predictor(dur_cond, x_masks=txt_tokens == PAD_INDEX, infer=infer)
            return encoder_out, ph_dur_pred
        else:
            return encoder_out, None


class MelodyEncoder(paddle.nn.Layer):
    def __init__(self, enc_hparams: dict):
        super().__init__()

        def get_hparam(key):
            return enc_hparams.get(key, hparams.get(key))

        hidden_size = get_hparam("hidden_size")
        self.note_midi_embed = Linear(1, hidden_size)
        self.note_dur_embed = Linear(1, hidden_size)
        self.use_glide_embed = hparams["use_glide_embed"]
        self.glide_embed_scale = hparams["glide_embed_scale"]
        if self.use_glide_embed:
            self.note_glide_embed = Embedding(len(hparams["glide_types"]) + 1, hidden_size, padding_idx=0)
        self.encoder = FastSpeech2Encoder(
            hidden_size=hidden_size,
            num_layers=get_hparam("enc_layers"),
            ffn_kernel_size=get_hparam("enc_ffn_kernel_size"),
            ffn_act=get_hparam("ffn_act"),
            dropout=get_hparam("dropout"),
            num_heads=get_hparam("num_heads"),
            use_pos_embed=get_hparam("use_pos_embed"),
            rel_pos=get_hparam("rel_pos"),
        )
        self.out_proj = Linear(hidden_size, hparams["hidden_size"])

    def forward(self, note_midi, note_rest, note_dur, glide=None):
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :param note_dur: int64 [B, T_n]
        :param glide: int64 [B, T_n]
        :return: [B, T_n, H]
        """
        midi_embed = self.note_midi_embed(note_midi[:, :, None]) * ~note_rest[:, :, None]
        dur_embed = self.note_dur_embed(note_dur.astype(dtype="float32")[:, :, None])
        ornament_embed = 0
        if self.use_glide_embed:
            ornament_embed += self.note_glide_embed(glide) * self.glide_embed_scale
        encoder_out = self.encoder(midi_embed, dur_embed + ornament_embed, padding_mask=note_midi < 0)
        encoder_out = self.out_proj(encoder_out)
        return encoder_out
