# Copyright 2023 The HuggingFace Team. All rights reserved.
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
Paddle utilities: Utilities related to Paddle
"""
import contextlib
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

from . import logging
from .import_utils import is_paddle_available

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# dummpy decorator, we do not use it
def maybe_allow_in_graph(cls):
    return cls


if is_paddle_available():
    import paddle

    class RNGStatesTracker:
        def __init__(self):
            self.states_ = {}
            self.mutex = threading.Lock()

        def reset(self):
            with self.mutex:
                self.states_ = {}

        def remove(self, generator_name=None):
            with self.mutex:
                if generator_name is not None:
                    del self.states_[generator_name]

        def manual_seed(self, seed, generator_name=None):
            with self.mutex:
                if generator_name is None:
                    generator_name = str(time.time())
                if generator_name in self.states_:
                    raise ValueError("state {} already exists".format(generator_name))
                orig_rng_state = paddle.get_cuda_rng_state()
                paddle.seed(seed)
                self.states_[generator_name] = paddle.get_cuda_rng_state()
                paddle.set_cuda_rng_state(orig_rng_state)
                return generator_name

        @contextlib.contextmanager
        def rng_state(self, generator_name=None):
            if generator_name is not None:
                if generator_name not in self.states_:
                    raise ValueError("state {} does not exist".format(generator_name))
                with self.mutex:
                    orig_cuda_rng_state = paddle.get_cuda_rng_state()
                    paddle.set_cuda_rng_state(self.states_[generator_name])
                    try:
                        yield
                    finally:
                        self.states_[generator_name] = paddle.get_cuda_rng_state()
                        paddle.set_cuda_rng_state(orig_cuda_rng_state)
            else:
                yield

    RNG_STATE_TRACKER = RNGStatesTracker()

    def get_rng_state_tracker(*args, **kwargs):
        return RNG_STATE_TRACKER

    paddle.Generator = get_rng_state_tracker

    randn = paddle.randn
    rand = paddle.rand
    randint = paddle.randint

    @paddle.jit.not_to_static
    def randn_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        is_bfloat16 = "bfloat16" in str(dtype) or "bfloat16" in paddle.get_default_dtype()
        if is_bfloat16:
            if generator is None:
                return randn(shape, dtype=paddle.bfloat16, name=name)
            else:
                with get_rng_state_tracker().rng_state(generator):
                    return randn(shape, dtype=paddle.bfloat16, name=name)
        else:
            if generator is None:
                return randn(shape, dtype=dtype, name=name)
            else:
                with get_rng_state_tracker().rng_state(generator):
                    return randn(shape, dtype=dtype, name=name)

    @paddle.jit.not_to_static
    def rand_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return rand(shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return rand(shape, dtype=dtype, name=name)

    @paddle.jit.not_to_static
    def randint_pt(low=0, high=None, shape=[1], dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return randint(low=low, high=high, shape=shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return randint(low=low, high=high, shape=shape, dtype=dtype, name=name)

    @paddle.jit.not_to_static
    def randn_like_pt(x, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if dtype is None:
            dtype = x.dtype
        return randn_pt(x.shape, dtype=dtype, generator=generator, name=name, **kwargs)

    paddle.randn = randn_pt
    paddle.rand = rand_pt
    paddle.randint = randint_pt
    paddle.randn_like = randn_like_pt

    def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["paddle.Generator"], "paddle.Generator"]] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """A helper function to create random tensors with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, (list, tuple)):
            batch_size = shape[0]
            shape = (1,) + tuple(shape[1:])
            latents = [randn_pt(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            latents = paddle.concat(latents, axis=0)
        else:
            latents = randn_pt(shape, generator=generator, dtype=dtype)

        return latents

    def rand_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["paddle.Generator"], "paddle.Generator"]] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """A helper function to create random tensors with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, (list, tuple)):
            batch_size = shape[0]
            shape = [
                1,
            ] + shape[1:]
            latents = [rand_pt(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            latents = paddle.concat(latents, axis=0)
        else:
            latents = rand_pt(shape, generator=generator, dtype=dtype)

        return latents

    def randint_tensor(
        low=0,
        high=None,
        shape: Union[Tuple, List] = [1],
        generator: Optional["paddle.Generator"] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
        will always be created on CPU.
        """
        latents = randint_pt(low=low, high=high, shape=shape, dtype=dtype, generator=generator)

        return latents

    if not hasattr(paddle, "dtype_guard"):

        @contextmanager
        def dtype_guard(dtype="float32"):
            origin_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype(dtype)
            try:
                yield
            finally:
                paddle.set_default_dtype(origin_dtype)

        paddle.dtype_guard = dtype_guard

    if not hasattr(paddle, "device_guard"):

        @contextmanager
        def device_guard(device="cpu", dev_id=0):
            device = device.replace("cuda", "gpu")
            if ":" in device:
                device, dev_id = device.split(":")
            origin_device = paddle.device.get_device()
            if device == "cpu":
                paddle.set_device(device)
            elif device in ["gpu", "xpu", "npu"]:
                paddle.set_device("{}:{}".format(device, dev_id))
            try:
                yield
            finally:
                paddle.set_device(origin_device)

        paddle.device_guard = device_guard

    _init_weights = True

    @contextmanager
    def no_init_weights(_enable=True):
        """
        Context manager to globally disable weight initialization to speed up loading large models.

        TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
        """
        global _init_weights
        old_init_weights = _init_weights
        if _enable:
            _init_weights = False
        try:
            yield
        finally:
            _init_weights = old_init_weights

    def is_compiled_module(module) -> bool:
        """Check whether the module was compiled with torch.compile()"""
        return False

    from paddle.fft import fftn, fftshift, ifftn, ifftshift

    def fourier_filter(x_in: paddle.Tensor, threshold: int, scale: int) -> paddle.Tensor:
        """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

        This version of the method comes from here:
        https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
        """
        x = x_in
        B, C, H, W = x.shape

        # Non-power of 2 images must be float32
        if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
            x = x.cast(dtype=paddle.float32)

        # FFT
        x_freq = fftn(x, axes=(-2, -1))
        x_freq = fftshift(x_freq, axes=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = paddle.ones((B, C, H, W))

        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
        x_freq = x_freq * mask

        # IFFT
        x_freq = ifftshift(x_freq, axes=(-2, -1))
        x_filtered = ifftn(x_freq, axes=(-2, -1)).real

        return x_filtered.cast(dtype=x_in.dtype)

    def apply_freeu(
        resolution_idx: int, hidden_states: paddle.Tensor, res_hidden_states: paddle.Tensor, **freeu_kwargs
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Applies the FreeU mechanism as introduced in https:
        //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

        Args:
            resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
            hidden_states (`paddle.Tensor`): Inputs to the underlying block.
            res_hidden_states (`paddle.Tensor`): Features from the skip block corresponding to the underlying block.
            s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
            s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if resolution_idx == 0:
            num_half_channels = hidden_states.shape[1] // 2
            hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b1"]
            res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s1"])
        if resolution_idx == 1:
            num_half_channels = hidden_states.shape[1] // 2
            hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b2"]
            res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s2"])

        return hidden_states, res_hidden_states

    def dim2perm(ndim, dim0, dim1):
        perm = list(range(ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return perm