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

from typing import TYPE_CHECKING

from ..utils import PPDIFFUSERS_SLOW_IMPORT, _LazyModule, is_paddle_available

_import_structure = {}

if is_paddle_available():
    _import_structure["adapter"] = ["MultiAdapter", "T2IAdapter"]
    _import_structure["autoencoder_asym_kl"] = ["AsymmetricAutoencoderKL"]
    _import_structure["autoencoder_kl"] = ["AutoencoderKL"]
    _import_structure["autoencoder_kl_temporal_decoder"] = ["AutoencoderKLTemporalDecoder"]
    _import_structure["autoencoder_tiny"] = ["AutoencoderTiny"]
    _import_structure["consistency_decoder_vae"] = ["ConsistencyDecoderVAE"]
    _import_structure["controlnet"] = ["ControlNetModel"]
    _import_structure["dual_transformer_2d"] = ["DualTransformer2DModel"]
    _import_structure["modeling_utils"] = ["ModelMixin"]
    _import_structure["prior_transformer"] = ["PriorTransformer"]
    _import_structure["t5_film_transformer"] = ["T5FilmDecoder"]
    _import_structure["transformer_2d"] = ["Transformer2DModel"]
    _import_structure["transformer_temporal"] = ["TransformerTemporalModel"]
    _import_structure["unet_1d"] = ["UNet1DModel"]
    _import_structure["unet_2d"] = ["UNet2DModel"]
    _import_structure["unet_2d_condition"] = ["UNet2DConditionModel"]
    _import_structure["unet_3d_condition"] = ["UNet3DConditionModel"]
    _import_structure["unet_kandi3"] = ["Kandinsky3UNet"]
    _import_structure["unet_motion_model"] = ["MotionAdapter", "UNetMotionModel"]
    _import_structure["unet_spatio_temporal_condition"] = ["UNetSpatioTemporalConditionModel"]
    _import_structure["vq_model"] = ["VQModel"]
    _import_structure["uvit_t2i"] = ["UViTT2IModel"]
    _import_structure["dit_llama"] = ["DiTLLaMA2DModel"]
    _import_structure["dit_llama_t2i"] = ["DiTLLaMAT2IModel"]
    # NOTE, new add
    _import_structure["lvdm_vae"] = ["LVDMAutoencoderKL"]
    _import_structure["lvdm_unet_3d"] = ["LVDMUNet3DModel"]
    _import_structure["ema"] = ["LitEma"]
    _import_structure["paddleinfer_runtime"] = ["PaddleInferRuntimeModel"]
    # NOTE, new add
    _import_structure["modelscope_autoencoder_img2vid"] = ["AutoencoderKL_imgtovideo"]
    _import_structure["modelscope_gaussian_diffusion"] = ["GaussianDiffusion"]
    _import_structure["modelscope_gaussion_sdedit"] = ["GaussianDiffusion_SDEdit"]
    _import_structure["modelscope_st_unet"] = ["STUNetModel"]
    _import_structure["modelscope_st_unet_video2video"] = ["Vid2VidSTUNet"]


if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    if is_paddle_available():
        from .adapter import MultiAdapter, T2IAdapter
        from .autoencoder_asym_kl import AsymmetricAutoencoderKL
        from .autoencoder_kl import AutoencoderKL
        from .autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
        from .autoencoder_tiny import AutoencoderTiny
        from .consistency_decoder_vae import ConsistencyDecoderVAE
        from .controlnet import ControlNetModel
        from .dit_llama import DiTLLaMA2DModel
        from .dit_llama_t2i import DiTLLaMAT2IModel
        from .dual_transformer_2d import DualTransformer2DModel

        # NOTE, new add
        from .ema import LitEma
        from .lvdm_unet_3d import LVDMUNet3DModel
        from .lvdm_vae import LVDMAutoencoderKL
        from .modeling_utils import ModelMixin
        from .modelscope_autoencoder_img2vid import AutoencoderKL_imgtovideo
        from .modelscope_gaussian_diffusion import GaussianDiffusion
        from .modelscope_gaussion_sdedit import GaussianDiffusion_SDEdit
        from .modelscope_st_unet import STUNetModel
        from .modelscope_st_unet_video2video import Vid2VidSTUNet
        from .paddleinfer_runtime import PaddleInferRuntimeModel
        from .prior_transformer import PriorTransformer
        from .t5_film_transformer import T5FilmDecoder
        from .transformer_2d import Transformer2DModel
        from .transformer_temporal import TransformerTemporalModel
        from .unet_1d import UNet1DModel
        from .unet_2d import UNet2DModel
        from .unet_2d_condition import UNet2DConditionModel
        from .unet_3d_condition import UNet3DConditionModel
        from .unet_kandi3 import Kandinsky3UNet
        from .unet_motion_model import MotionAdapter, UNetMotionModel
        from .unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
        from .uvit_t2i import UViTT2IModel
        from .vq_model import VQModel
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
