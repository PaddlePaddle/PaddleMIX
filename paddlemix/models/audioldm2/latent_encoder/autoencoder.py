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

import paddle
import paddle.nn as nn
import os

import paddle.nn.functional as F
import numpy as np

from ..unet.model import Encoder, Decoder
import soundfile as sf

from ..hifigan.model import get_vocoder, synth_one_sample

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = paddle.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * paddle.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        else:
            if other is None:
                return 0.5 * paddle.mean(
                    paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * paddle.mean(
                    paddle.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * paddle.sum(
            logtwopi + self.logvar + paddle.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, paddle.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for paddle.exp().
    logvar1, logvar2 = [
        x if isinstance(x, paddle.Tensor) else paddle.to_tensor(x, dtype=tensor.dtype)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


class AutoencoderKL(nn.Layer):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        sampling_rate=16000,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()
        self.automatic_optimization = False
        assert (
            "mel_bins" in ddconfig.keys()
        ), "mel_bins is not specified in the Autoencoder config"
        num_mel = ddconfig["mel_bins"]
        self.image_key = image_key
        self.sampling_rate = sampling_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.loss = None
        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2D(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2D(embed_dim, ddconfig["z_channels"], 1)

        if self.image_key == "fbank":
            self.vocoder = get_vocoder(None, num_mel)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", paddle.randn([3, colorize_nlabels, 1, 1]))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.learning_rate = float(base_learning_rate)
        # print("Initial learning rate %s" % self.learning_rate)

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_exp_name = None

    def get_log_dir(self):
        if self.logger_save_dir is None and self.logger_exp_name is None:
            return os.path.join(self.logger.save_dir, self.logger._project)
        else:
            return os.path.join(self.logger_save_dir, self.logger_exp_name)

    def set_log_dir(self, save_dir, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_name = exp_name

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = paddle.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_dict(sd)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_to_waveform(self, dec):
        from audioldm2.hifigan.model import vocoder_infer

        if self.image_key == "fbank":
            dec = dec.squeeze(1).transpose([0, 2, 1])
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).transpose([0, 2, 1])
            wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction

    def visualize_latent(self, input):
        import matplotlib.pyplot as plt

        np.save("input.npy", input.detach().numpy())
        # zero_input = paddle.zeros_like(input) - 11.59
        time_input = input.clone()
        time_input[:, :, :, :32] *= 0
        time_input[:, :, :, :32] -= 11.59

        np.save("time_input.npy", time_input.detach().numpy())

        posterior = self.encode(time_input)
        latent = posterior.sample()
        np.save("time_latent.npy", latent.detach().numpy())
        avg_latent = paddle.mean(latent, axis=1)
        for i in range(avg_latent.shape[0]):
            plt.imshow(avg_latent[i].detach().numpy().T)
            plt.savefig("freq_%s.png" % i)
            plt.close()

        freq_input = input.clone()
        freq_input[:, :, :512, :] *= 0
        freq_input[:, :, :512, :] -= 11.59

        np.save("freq_input.npy", freq_input.detach().numpy())

        posterior = self.encode(freq_input)
        latent = posterior.sample()
        np.save("freq_latent.npy", latent.detach().numpy())
        avg_latent = paddle.mean(latent, axis=1)
        for i in range(avg_latent.shape[0]):
            plt.imshow(avg_latent[i].detach().numpy().T)
            plt.savefig("time_%s.png" % i)
            plt.close()

    def get_input(self, batch):
        fname, text, label_indices, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["label_vector"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )

        ret = {}

        ret["fbank"], ret["stft"], ret["fname"], ret["waveform"] = (
            fbank.unsqueeze(1),
            stft.unsqueeze(1),
            fname,
            waveform.unsqueeze(1),
        )

        return ret

    def save_wave(self, batch_wav, fname, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for wav, name in zip(batch_wav, fname):
            name = os.path.basename(name)

            sf.write(os.path.join(save_dir, name), wav, samplerate=self.sampling_rate)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @paddle.no_grad()
    def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
        log = dict()
        x = batch
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(posterior.sample())
            log["reconstructions"] = xrec

        log["inputs"] = x
        wavs = self._log_img(log, train=train, index=0, waveform=waveform)
        return wavs

    def _log_img(self, log, train=True, index=0, waveform=None):
        images_input = self.tensor2numpy(log["inputs"][index, 0]).T
        images_reconstruct = self.tensor2numpy(log["reconstructions"][index, 0]).T
        images_samples = self.tensor2numpy(log["samples"][index, 0]).T

        if train:
            name = "train"
        else:
            name = "val"

        if self.logger is not None:
            self.logger.log_image(
                "img_%s" % name,
                [images_input, images_reconstruct, images_samples],
                caption=["input", "reconstruct", "samples"],
            )

        inputs, reconstructions, samples = (
            log["inputs"],
            log["reconstructions"],
            log["samples"],
        )

        if self.image_key == "fbank":
            wav_original, wav_prediction = synth_one_sample(
                inputs[index],
                reconstructions[index],
                labels="validation",
                vocoder=self.vocoder,
            )
            wav_original, wav_samples = synth_one_sample(
                inputs[index], samples[index], labels="validation", vocoder=self.vocoder
            )
            wav_original, wav_samples, wav_prediction = (
                wav_original[0],
                wav_samples[0],
                wav_prediction[0],
            )
        elif self.image_key == "stft":
            wav_prediction = (
                self.decode_to_waveform(reconstructions)[index, 0]
                .detach()
                .numpy()
            )
            wav_samples = (
                self.decode_to_waveform(samples)[index, 0].detach().numpy()
            )
            wav_original = waveform[index, 0].detach().numpy()

        return wav_original, wav_prediction, wav_samples

    def tensor2numpy(self, tensor):
        return tensor.detach().numpy()

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", paddle.randn([3, x.shape[1], 1, 1], dtype=x.dtype))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
    