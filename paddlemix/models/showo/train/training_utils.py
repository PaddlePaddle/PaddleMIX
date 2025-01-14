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

import copy
import random
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import paddle
import pandas as pd


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)


class EMA:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[paddle.base.framework.EagerParamBase.from_tensor],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.temp_stored_params = None
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None
        self.model_cls = model_cls
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "EMA":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)
        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config)
        ema_model.set_state_dict(state_dict=ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")
        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")
        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)
        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)
        cur_decay_value = min(cur_decay_value, self.decay)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @paddle.no_grad()
    def step(self, parameters: Iterable[paddle.base.framework.EagerParamBase.from_tensor]):
        parameters = list(parameters)
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay
        for s_param, param in zip(self.shadow_params, parameters):
            if not param.stop_gradient:
                s_param.subtract_(y=paddle.to_tensor(one_minus_decay * (s_param - param)))
            else:
                paddle.assign(param, output=s_param)
        paddle.device.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[paddle.base.framework.EagerParamBase.from_tensor]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.place).data)

    def to(self, device=None, dtype=None) -> None:
        """Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        self.shadow_params = [
            (p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device))
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        """
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def store(self, parameters: Iterable[paddle.base.framework.EagerParamBase.from_tensor]) -> None:
        """
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[paddle.base.framework.EagerParamBase.from_tensor]) -> None:
        """
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")
        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")
        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")
        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")
        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")
        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, paddle.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")


def pixel_entropy_per_percent_masked_bucket(logits, input_ids, mask_id):
    masked_tokens = input_ids == mask_id
    num_masked_pixels = masked_tokens.sum(axis=-1)
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    log_probs = paddle.nn.functional.log_softmax(x=logits, axis=-1)
    entropy_per_pixel = -(probs * log_probs).sum(axis=-1)
    entropy_per_pixel[~masked_tokens] = 0
    entropy_per_image_numerator = entropy_per_pixel.sum(axis=-1)
    entropy_per_image = entropy_per_image_numerator / num_masked_pixels
    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)
    entropy_by_masked_bucket = average_by_buckets(entropy_per_image, masked_buckets, total_buckets)
    return entropy_by_masked_bucket


def image_entropy_per_percent_masked_bucket(logits, input_ids, mask_id):
    masked_tokens = input_ids == mask_id
    num_masked_pixels = masked_tokens.sum(axis=-1, keepdim=True)
    pixel_probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    pixel_probs[~masked_tokens] = 0
    image_probs_numerator = pixel_probs.sum(axis=-2)
    image_probs = image_probs_numerator / num_masked_pixels
    image_log_probs = image_probs.log()
    entropy_per_image = -(image_probs * image_log_probs).sum(axis=-1)
    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)
    entropy_by_masked_bucket = average_by_buckets(entropy_per_image, masked_buckets, total_buckets)
    return entropy_by_masked_bucket


def cross_entropy_per_percent_masked_bucket(logits, labels, input_ids, mask_id, output_size, label_smoothing):
    cross_entropy_per_image = paddle.nn.functional.cross_entropy(
        input=logits.reshape([-1, output_size]),
        label=labels.reshape([-1]),
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)
    cross_entropy_by_percent_masked_bucket = average_by_buckets(cross_entropy_per_image, masked_buckets, total_buckets)
    return cross_entropy_by_percent_masked_bucket


def token_probability_distributions_per_percent_masked_bucket(logits, input_ids, mask_id):
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)
    data = []
    for bucket_idx in range(total_buckets):
        indices_for_bucket = masked_buckets[masked_buckets == bucket_idx]
        if tuple(indices_for_bucket.shape)[0] == 0:
            continue
        index_for_bucket = indices_for_bucket[0]
        image_probs = probs[index_for_bucket]
        input_ids_for_image = input_ids[index_for_bucket]
        masked_pixels_probs = image_probs[input_ids_for_image == mask_id]
        masked_pixel_probs = masked_pixels_probs[0]
        masked_pixel_probs = masked_pixel_probs.cpu().numpy()
        for masked_pixel_prob in masked_pixel_probs:
            data.append({"bucket": bucket_idx, "masked_pixel_prob": masked_pixel_prob})
    df = pd.DataFrame(data)
    return df


def average_by_buckets(values, masked_buckets, total_buckets):
    unique_buckets, bucket_counts = masked_buckets.unique(axis=0, return_counts=True)
    numerator = paddle.zeros(shape=total_buckets)
    numerator.put_along_axis_(axis=0, indices=masked_buckets, values=values, reduce="add")
    denominator = paddle.ones(shape=total_buckets, dtype="int64")
    denominator[unique_buckets] = bucket_counts
    averaged_by_buckets = numerator / denominator
    return averaged_by_buckets


def input_ids_to_masked_buckets(input_ids, mask_id, total_buckets=10):
    assert total_buckets == 10
    masked_percent = (input_ids == mask_id).sum(axis=-1) / tuple(input_ids.shape)[-1]
    masked_buckets = (
        ((0 < masked_percent) & (masked_percent <= 0.1)) * 0
        + ((0.1 < masked_percent) & (masked_percent <= 0.2)) * 1
        + ((0.2 < masked_percent) & (masked_percent <= 0.3)) * 2
        + ((0.3 < masked_percent) & (masked_percent <= 0.4)) * 3
        + ((0.4 < masked_percent) & (masked_percent <= 0.5)) * 4
        + ((0.5 < masked_percent) & (masked_percent <= 0.6)) * 5
        + ((0.6 < masked_percent) & (masked_percent <= 0.7)) * 6
        + ((0.7 < masked_percent) & (masked_percent <= 0.8)) * 7
        + ((0.8 < masked_percent) & (masked_percent <= 0.9)) * 8
        + ((0.9 < masked_percent) & (masked_percent <= 1.0)) * 9
    )
    return masked_buckets
