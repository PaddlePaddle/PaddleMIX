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
import paddle.nn as nn
import os
import numpy as np
from paddlemix.models.model_utils import MixPretrainedModel
# from ppdiffusers.models import LitEma
import soundfile as sf
import tqdm
from .encoders.clap_encoder import CLAPAudioEmbeddingClassifierFreev2
from .latentdiffusion_samplers import DDIMSampler, PLMSSampler
from .latent_encoder.autoencoder import DiagonalGaussianDistribution
from .diffusionwrapper import (
    DiffusionWrapper,
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    default,
    instantiate_from_config,
    disabled_train
)
from .configuration import AudioLDM2Config

__all__ = [
    "AudioLDM2Model",
    "AudioLDM2PretrainedModel",
]

class AudioLDM2PretrainedModel(MixPretrainedModel):
    """
    The class for pretrained model of AudioLDM2.
    """

    model_config_file = "config.json"
    config_class = AudioLDM2Config
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "audioldm2"

class AudioLDM2Model(AudioLDM2PretrainedModel):
    """
    Args:
        config (:class:`AudioLDM2Config`):
    """

    def __init__(self, config: AudioLDM2Config):
        super(AudioLDM2Model, self).__init__(config)
        assert config.parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = config.parameterization
        self.device_name = config.device

        self.clip_denoised = False
        self.log_every_t = config.log_every_t
        self.first_stage_key = config.first_stage_key
        self.sampling_rate = config.sampling_rate
        # self.use_ema = True
        # if self.use_ema:
        #     self.model_ema = LitEma(self.model)

        self.clap = CLAPAudioEmbeddingClassifierFreev2(
            pretrained_path="",
            enable_cuda=self.device_name=="gpu",
            sampling_rate=self.sampling_rate,
            embed_mode="audio",
            amodel="HTSAT-base",
        )
        self.latent_t_size = config.latent_t_size
        self.latent_f_size = config.latent_f_size
        self.channels = config.channels
        self.use_positional_encodings = False
        self.conditioning_key = list(config.cond_stage_config.keys())
        self.model = DiffusionWrapper(config.unet_config, self.conditioning_key)

        self.v_posterior = 0.0  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta

        self.num_timesteps_cond = default(config.num_timesteps_cond, 1)
        assert self.num_timesteps_cond <= config.timesteps
        self.register_schedule(
            beta_schedule="linear",
            timesteps=config.timesteps,
            linear_start=config.linear_start,
            linear_end=config.linear_end,
            cosine_s=8e-3,
        )
        logvar_init = 0.0
        self.logvar = paddle.full(shape=(self.num_timesteps,), fill_value=logvar_init)
        self.logvar = paddle.create_parameter(
            shape=self.logvar.shape,
            dtype=str(self.logvar.numpy().dtype),
            default_initializer=nn.initializer.Assign(self.logvar)
        )
        self.logvar.stop_gradient = True

        self.register_buffer("scale_factor", paddle.to_tensor(1.0))
        self.instantiate_first_stage(config.first_stage_config)
        self.unconditional_prob_cfg = config.unconditional_prob_cfg
        self.cond_stage_models = nn.LayerList([])
        self.instantiate_cond_stage(config.cond_stage_config)
        self.conditional_dry_run_finished = False

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.stop_gradient = True

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}
        for i, cond_model_key in enumerate(config.keys()):
            if "params" in config[cond_model_key] and "device" in config[cond_model_key]["params"]:
                config[cond_model_key]["params"]["device"] = self.device_name
            model = instantiate_from_config(config[cond_model_key])
            model = model.to(self.device_name)
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }
    
    def make_cond_schedule(
        self,
    ):
        self.cond_ids = paddle.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype="int64",
        )
        ids = paddle.cast(
            paddle.round(
                paddle.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
            ),
            dtype="int64"
        )
        self.cond_ids[: self.num_timesteps_cond] = ids


    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        self.register_buffer("betas", paddle.to_tensor(betas, dtype="float32"))
        self.register_buffer("alphas_cumprod", paddle.to_tensor(alphas_cumprod, dtype="float32"))
        self.register_buffer("alphas_cumprod_prev", paddle.to_tensor(alphas_cumprod_prev, dtype="float32"))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", paddle.to_tensor(np.sqrt(alphas_cumprod), dtype="float32"))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", paddle.to_tensor(np.sqrt(1.0 - alphas_cumprod), dtype="float32")
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", paddle.to_tensor(np.log(1.0 - alphas_cumprod), dtype="float32")
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", paddle.to_tensor(np.sqrt(1.0 / alphas_cumprod), dtype="float32")
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", paddle.to_tensor(np.sqrt(1.0 / alphas_cumprod - 1), dtype="float32")
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", paddle.to_tensor(posterior_variance, dtype="float32"))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            paddle.to_tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype="float32"),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            paddle.to_tensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype="float32"),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            paddle.to_tensor(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
                dtype="float32"
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * paddle.to_tensor(alphas, dtype="float32")
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(paddle.to_tensor(alphas_cumprod, dtype="float32"))
                / (2.0 * 1 - paddle.to_tensor(alphas_cumprod, dtype="float32"))
            )
        elif self.parameterization == "v":
            lvlb_weights = paddle.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * paddle.to_tensor(alphas, dtype="float32")
                    * (1 - self.alphas_cumprod)
                )
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistable=False)
        assert not paddle.isnan(self.lvlb_weights).all()

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def make_decision(self, probability):
        if float(paddle.rand([])) < probability:
            return True
        else:
            return False

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clip_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @paddle.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_ = x.shape
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (
            (1 - paddle.cast(t == 0, "float32")).reshape((b, *((1,) * (len(x.shape) - 1))))
        )

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @paddle.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        b = shape[0]
        if x_T is None:
            img = paddle.randn(shape)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = paddle.full((b,), i, dtype="int64")

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts]
                cond = self.q_sample(x_start=cond, t=tc, noise=paddle.randn(cond.shapes))

            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )

            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @paddle.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            **kwargs,
        )
    
    @paddle.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.shape[-2], mask.shape[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim and not use_plms:
            ddim_sampler = DDIMSampler(self, device=self.device_name)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )
        elif use_plms:
            plms_sampler = PLMSSampler(self)
            samples, intermediates = plms_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        else:
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: paddle.randn(x_start.shape))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    def _get_input(self, batch, k):
        fname, text, waveform, stft, fbank, phoneme_idx = (
            batch["fname"],
            batch["text"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
            batch["phoneme_idx"]
        )
        ret = {}

        ret["fbank"] = (
            paddle.cast(fbank.unsqueeze(1), dtype="float32")
        )
        ret["stft"] = paddle.cast(stft, dtype="float32")
        ret["waveform"] = paddle.cast(waveform, dtype="float32")
        ret["phoneme_idx"] = paddle.cast(phoneme_idx, dtype="int64")
        ret["text"] = list(text)
        ret["fname"] = fname

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.sample()
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, paddle.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z
    
    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ](c)
        else:
            # when the cond_stage_key is "all", pick one random element out
            if isinstance(c, dict):
                c = c[list(c.keys())[0]]

            if isinstance(c, paddle.Tensor):
                batchsize = c.shape[0]
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()

            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ].get_unconditional_condition(batchsize)

        return c
    
    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_decoding_output=False,
        return_encoder_input=False,
        return_encoder_output=False,
        unconditional_prob_cfg=0.1,
    ):
        x = self._get_input(batch, k)

        if return_first_stage_encode:
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
        else:
            z = None
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False
            if self.conditional_dry_run_finished and self.make_decision(
                unconditional_prob_cfg
            ):
                unconditional_cfg = True
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key][
                    "cond_stage_key"
                ]

                if cond_model_key in cond_dict.keys():
                    continue

                # The original data for conditioning
                # If cond_model_key is "all", that means the conditional model need all the information from a batch
                if cond_stage_key != "all":
                    xc = self._get_input(batch, cond_stage_key)
                else:
                    xc = batch
                # if cond_stage_key is "all", xc will be a dictionary containing all keys
                # Otherwise xc will be an entry of the dictionary
                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )
                # cond_dict will be used to condition the diffusion model
                # If one conditional model return multiple conditioning signal
                if isinstance(c, dict):
                    for k in c.keys():
                        cond_dict[k] = c[k]
                else:
                    cond_dict[cond_model_key] = c
                
        out = [z, cond_dict]

        if return_decoding_output:
            xrec = self.decode_first_stage(z)
            out += [xrec]

        if return_encoder_input:
            out += [x]

        if return_encoder_output:
            out += [encoder_posterior]

        if not self.conditional_dry_run_finished:
            self.conditional_dry_run_finished = True

        # Output is a dictionary, where the value could only be tensor or tuple
        return out

    def encode_first_stage(self, x):
        with paddle.no_grad():
            return self.first_stage_model.encode(x)
        
    def decode_first_stage(self, z):
        with paddle.no_grad():
            z = 1.0 / self.scale_factor * z
            decoding = self.first_stage_model.decode(z)
        return decoding
    
    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.shape) == 4:
            mel = mel.squeeze(1)
        mel = mel.transpose([0, 2, 1])
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform
    
    def save_waveform(self, waveform, savepath, name="outwav"):
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            todo_waveform = waveform[i, 0]
            todo_waveform = (
                todo_waveform / np.max(np.abs(todo_waveform))
            ) * 0.8  # Normalize the energy of the generation output
            sf.write(path, todo_waveform, samplerate=self.sampling_rate)

    def filter_useful_cond_dict(self, cond_dict):
        new_cond_dict = {}
        for key in cond_dict.keys():
            if key in self.cond_stage_model_metadata.keys():
                new_cond_dict[key] = cond_dict[key]

        # All the conditional key in the metadata should be used
        for key in self.cond_stage_model_metadata.keys():
            assert key in new_cond_dict.keys(), "%s, %s" % (
                key,
                str(new_cond_dict.keys()),
            )

        return new_cond_dict
    
    def reorder_cond_dict(self, cond_dict):
        # To make sure the order is correct
        new_cond_dict = {}
        for key in self.conditioning_key:
            new_cond_dict[key] = cond_dict[key]
        return new_cond_dict

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        cond = self.reorder_cond_dict(cond)

        x_recon = self.model(x_noisy, t, cond_dict=cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    
    def forward(
        self,
        batch,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        assert x_T is None

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None

        # with self.ema_scope("Plotting"):
        for i in range(1):
            z, c = self.get_input(
                batch,
                self.first_stage_key,
                unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
            )

            c = self.filter_useful_cond_dict(c)

            text = self._get_input(batch, "text")

            # Generate multiple samples
            batch_size = z.shape[0] * n_gen

            # Generate multiple samples at a time and filter out the best
            # The condition to the diffusion wrapper can have many format
            for cond_key in c.keys():
                if isinstance(c[cond_key], list):
                    for i in range(len(c[cond_key])):
                        c[cond_key][i] = paddle.concat([c[cond_key][i]] * n_gen, axis=0)
                elif isinstance(c[cond_key], dict):
                    for k in c[cond_key].keys():
                        c[cond_key][k] = paddle.concat([c[cond_key][k]] * n_gen, axis=0)
                else:
                    c[cond_key] = paddle.concat([c[cond_key]] * n_gen, axis=0)

            text = text * n_gen

            if unconditional_guidance_scale != 1.0:
                unconditional_conditioning = {}
                for key in self.cond_stage_model_metadata:
                    model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                    unconditional_conditioning[key] = self.cond_stage_models[
                        model_idx
                    ].get_unconditional_condition(batch_size)

            fnames = list(self._get_input(batch, "fname"))
            samples, _ = self.sample_log(
                cond=c,
                batch_size=batch_size,
                x_T=x_T,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                use_plms=use_plms,
            )

            mel = self.decode_first_stage(samples)

            waveform = self.mel_spectrogram_to_waveform(
                mel, savepath="", bs=None, name=fnames, save=False
            )

            if n_gen > 1:
                best_index = []
                similarity = self.clap.cos_similarity(
                    paddle.to_tensor(waveform, dtype="float32").squeeze(1), text
                )
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = paddle.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text:")
                print(' '.join('{:.2f}'.format(num) for num in similarity.detach().numpy().tolist()))
                print("Choose the following indexes as the output:", best_index)

            return waveform
