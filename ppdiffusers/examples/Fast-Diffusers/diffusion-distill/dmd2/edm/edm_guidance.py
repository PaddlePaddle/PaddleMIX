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

# code is heavily based on https://github.com/tianweiy/DMD2

import copy
import glob
import hashlib
import html
import io
import math
import os
import pickle
import re
import urllib
import urllib.request
import uuid
from typing import Any

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import requests
from edm.edm_network import get_edm_network


def open_url(
    url: str,
    cache_dir: str = None,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
    cache: bool = True,
) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    # assert is_url(url)

    # Lookup from cache.
    # if cache_dir is None:
    #     cache_dir = make_cache_dir_path("downloads")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link) for link in content_str.split('"') if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[: min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = paddle.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def _calculate_correct_fan(tensor, mode, reverse=False):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse)

    return fan_in if mode == "fan_in" else fan_out


def _calculate_fan_in_and_fan_out(tensor, reverse=False):
    """
    Calculate (fan_in, _fan_out) for tensor
    Args:
        tensor (Tensor): paddle.Tensor
        reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
            e.g. : conv.weight [cout, cin, kh, kw] is False; linear.weight [cin, cout] is True
    Return:
        Tuple[fan_in, fan_out]
    """
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if reverse:
        num_input_fmaps, num_output_fmaps = tensor.shape[0], tensor.shape[1]
    else:
        num_input_fmaps, num_output_fmaps = tensor.shape[1], tensor.shape[0]

    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_gain(nonlinearity, param=None):
    linear_fns = ["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(shape=tensor.shape, dtype=tensor.dtype, min=a, max=b))
    return tensor


def _no_grad_normal_(tensor, mean, std):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean, std, shape=tensor.shape))
    return tensor


def kaiming_uniform_init(param, a=0, mode="fan_in", nonlinearity="leaky_relu", reverse=False):
    """
    Modified tensor inspace using kaiming_uniform method
    Args:
        param (paddle.Tensor): paddle Tensor
        mode (str): ['fan_in', 'fan_out'], 'fin_in' defalut
        nonlinearity (str): nonlinearity method name
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan = _calculate_correct_fan(param, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(param, -k, k)


def reset_parameters(m, reverse=False):
    if not hasattr(m, "weight"):
        return
    if m.weight.ndim < 2:
        return

    if isinstance(m, nn.Linear):
        reverse = True

    kaiming_uniform_init(m.weight, a=math.sqrt(5), reverse=reverse)
    if m.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(m.weight, reverse=reverse)
        bound = 1 / math.sqrt(fan_in)
        _no_grad_uniform_(m.bias, -bound, bound)


class EDMGuidance(nn.Layer):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator

        with open_url(args.model_id) as f:
            temp_edm_state_dict = pickle.load(f)  # ['ema']

        # initialize the real unet
        self.real_unet = get_edm_network(args)
        # self.real_unet.load_state_dict(temp_edm.state_dict(), strict=True)
        self.real_unet.set_state_dict(temp_edm_state_dict)
        self.real_unet.requires_grad_(False)
        del self.real_unet.model.map_augment
        self.real_unet.model.map_augment = None

        # initialize the fake unet
        self.fake_unet = copy.deepcopy(self.real_unet)
        self.fake_unet.requires_grad_(True)

        # some training hyper-parameters
        self.sigma_data = args.sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        self.gan_classifier = args.gan_classifier
        self.diffusion_gan = args.diffusion_gan
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep

        if self.gan_classifier:
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2D(kernel_size=4, in_channels=768, out_channels=768, stride=2, padding=1),  # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=768),
                nn.Silu(),
                nn.Conv2D(kernel_size=4, in_channels=768, out_channels=768, stride=4, padding=0),  # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=768),
                nn.Silu(),
                nn.Conv2D(kernel_size=1, in_channels=768, out_channels=1, stride=1, padding=0),  # 1x1 -> 1x1
            )

            self.cls_pred_branch.apply(reset_parameters)
            self.cls_pred_branch.requires_grad_(True)

        self.num_train_timesteps = args.num_train_timesteps
        # small sigma first, large sigma later
        karras_sigmas = paddle.flip(
            get_sigmas_karras(
                self.num_train_timesteps, sigma_max=self.sigma_max, sigma_min=self.sigma_min, rho=self.rho
            ),
            axis=[0],
        )
        self.register_buffer("karras_sigmas", karras_sigmas)

        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)
        # del temp_edm

    def compute_distribution_matching_loss(self, latents, labels):
        original_latents = latents
        batch_size = latents.shape[0]

        with paddle.no_grad():
            timesteps = paddle.randint(
                self.min_step,
                min(self.max_step + 1, self.num_train_timesteps),
                [batch_size, 1, 1, 1],
                dtype=paddle.int64,
            )

            noise = paddle.randn_like(latents)

            timestep_sigma = self.karras_sigmas[timesteps]

            noisy_latents = latents + timestep_sigma.reshape([-1, 1, 1, 1]) * noise

            pred_real_image = self.real_unet(noisy_latents, timestep_sigma, labels)

            pred_fake_image = self.fake_unet(noisy_latents, timestep_sigma, labels)

            p_real = latents - pred_real_image
            p_fake = latents - pred_fake_image

            weight_factor = paddle.abs(p_real).mean(axis=[1, 2, 3], keepdim=True)
            grad = (p_real - p_fake) / weight_factor

            grad = paddle.nan_to_num(grad)

        # this loss gives the grad as gradient through autodiff, following https://github.com/ashawkey/stable-dreamfusion
        loss = 0.5 * F.mse_loss(original_latents, (original_latents - grad).detach(), reduction="mean")

        loss_dict = {"loss_dm": loss}

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_pred_real_image": pred_real_image.detach(),
            "dmtrain_pred_fake_image": pred_fake_image.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": paddle.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
        }
        return loss_dict, dm_log_dict

    def compute_loss_fake(
        self,
        latents,
        labels,
    ):
        batch_size = latents.shape[0]

        latents = latents.detach()  # no gradient to generator

        noise = paddle.randn_like(latents)

        timesteps = paddle.randint(0, self.num_train_timesteps, [batch_size, 1, 1, 1], dtype=paddle.int64)
        timestep_sigma = self.karras_sigmas[timesteps]
        noisy_latents = latents + timestep_sigma.reshape([-1, 1, 1, 1]) * noise

        fake_x0_pred = self.fake_unet(noisy_latents, timestep_sigma, labels)

        snrs = timestep_sigma**-2

        # weight_schedule karras
        weights = snrs + 1.0 / self.sigma_data**2

        target = latents

        loss_fake = paddle.mean(weights * (fake_x0_pred - target) ** 2)

        loss_dict = {"loss_fake_mean": loss_fake}

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach(),
        }
        return loss_dict, fake_log_dict

    def compute_cls_logits(self, image, label):
        if self.diffusion_gan:
            timesteps = paddle.randint(0, self.diffusion_gan_max_timestep, [image.shape[0]], dtype=paddle.int64)
            timestep_sigma = self.karras_sigmas[timesteps]
            noise_tmp = paddle.randn_like(image)
            image = image + timestep_sigma.reshape([-1, 1, 1, 1]) * noise_tmp
        else:
            timesteps = paddle.zeros([image.shape[0]], dtype=paddle.int64)
            timestep_sigma = self.karras_sigmas[timesteps]

        rep = self.fake_unet(image, timestep_sigma, label, return_bottleneck=True).to(dtype=paddle.float32)

        logits = self.cls_pred_branch(rep).squeeze(axis=[2, 3])

        return logits

    def compute_generator_clean_cls_loss(self, fake_image, fake_labels):
        loss_dict = {}

        pred_realism_on_fake_with_grad = self.compute_cls_logits(image=fake_image, label=fake_labels)
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss_dict

    def compute_guidance_clean_cls_loss(self, real_image, fake_image, real_label, fake_label):
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(),
            real_label,
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(),
            fake_label,
        )
        classification_loss = F.softplus(pred_realism_on_fake) + F.softplus(-pred_realism_on_real)

        log_dict = {
            "pred_realism_on_real": F.sigmoid(pred_realism_on_real).squeeze(axis=1).detach(),
            "pred_realism_on_fake": F.sigmoid(pred_realism_on_fake).squeeze(axis=1).detach(),
        }

        loss_dict = {"guidance_cls_loss": classification_loss.mean()}
        return loss_dict, log_dict

    def generator_forward(self, image, labels):
        loss_dict = {}
        log_dict = {}

        # image.requires_grad_(True)
        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(image, labels)

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        if self.gan_classifier:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(image, labels)
            loss_dict.update(clean_cls_loss_dict)

        return loss_dict, log_dict

    def guidance_forward(self, image, labels, real_train_dict=None):
        fake_dict, fake_log_dict = self.compute_loss_fake(image, labels)

        loss_dict = fake_dict
        log_dict = fake_log_dict

        if self.gan_classifier:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict["real_image"],
                fake_image=image,
                real_label=real_train_dict["real_label"],
                fake_label=labels,
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)
        return loss_dict, log_dict

    def forward(self, generator_turn=False, guidance_turn=False, generator_data_dict=None, guidance_data_dict=None):
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"], labels=generator_data_dict["label"]
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                labels=guidance_data_dict["label"],
                real_train_dict=guidance_data_dict["real_train_dict"],
            )
        else:
            raise NotImplementedError

        return loss_dict, log_dict
