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

import os
from typing import Union

from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.utils.log import logger

__all__ = ["AudioLDM2Config"]


class AudioLDM2Config(PretrainedConfig):

    model_type = "audioldm2"

    def __init__(
        self,
        model_name: str = "audioldm2-full",
        first_stage_key: str = "fbank",
        sampling_rate: int = 16000,
        parameterization: str = "eps",
        log_every_t: int = 200,
        latent_t_size: int = 256,
        latent_f_size: int = 16,
        channels: int = 8,
        timesteps: int = 1000,
        num_timesteps_cond: int = 1,
        linear_start: float = 0.0015,
        linear_end: float = 0.0195,
        unconditional_prob_cfg: float = 0.1,
        device: str = "gpu",
        unet_image_size: int = 64,
        unet_context_dim: list = [768, 1024],
        unet_in_channels: int = 8,
        unet_out_channels: int = 8,
        unet_model_channels: int = 128,
        unet_attention_resolutions: list = [8, 4, 2],
        unet_num_res_blocks: int = 2,
        unet_channel_mult: list = [1, 2, 3, 5],
        unet_num_head_channels: int = 32,
        unet_use_spatial_transformer: bool = True,
        unet_transformer_depth: int = 1,
        autoencoder_sampling_rate: int = 16000,
        autoencoder_batchsize: int = 4,
        autoencoder_image_key: str = "fbank",
        autoencoder_subband: int = 1,
        autoencoder_embed_dim: int = 8,
        autoencoder_time_shuffle: int = 1,
        ddconfig_double_z: bool = True,
        ddconfig_mel_bins: int = 64,
        ddconfig_z_channels: int = 8,
        ddconfig_resolution: int = 256,
        ddconfig_downsample_time: bool = False,
        ddconfig_in_channels: int = 1,
        ddconfig_out_ch: int = 1,
        ddconfig_ch: int = 128,
        ddconfig_ch_mult: list = [1, 2, 4],
        ddconfig_num_res_blocks: int = 2,
        ddconfig_attn_resolutions: list = [],
        ddconfig_dropout: float = 0.0,
        sequence2audiomae_always_output_audiomae_gt: bool = False,
        sequence2audiomae_learnable: bool = True,
        sequence2audiomae_use_gt_mae_output: bool = True,
        sequence2audiomae_use_gt_mae_prob: float = 0.0,
        sequence2audiomae_base_learning_rate: float = 0.0002,
        sequence2audiomae_sequence_gen_length: int = 8,
        sequence2audiomae_use_warmup: bool = True,
        sequence2audiomae_sequence_input_key: list = ["film_clap_cond1", "crossattn_flan_t5"],
        sequence2audiomae_sequence_input_embed_dim: list = [512, 1024],
        sequence2audiomae_batchsize: int = 16,
        sequence2audiomae_cond_stage_configs: dict = None,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.first_stage_key = first_stage_key
        self.sampling_rate = sampling_rate
        self.parameterization = parameterization
        self.log_every_t = log_every_t
        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size
        self.channels = channels
        self.timesteps = timesteps
        self.num_timesteps_cond = num_timesteps_cond
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.unconditional_prob_cfg = unconditional_prob_cfg
        self.device = device

        self.unet_config = {}
        self.unet_config["target"] = ".unet.openaimodel.UNetModel"
        self.unet_config["params"] = {}
        self.unet_config["params"]["image_size"] = unet_image_size
        self.unet_config["params"]["context_dim"] = unet_context_dim
        self.unet_config["params"]["in_channels"] = unet_in_channels
        self.unet_config["params"]["out_channels"] = unet_out_channels
        self.unet_config["params"]["model_channels"] = unet_model_channels
        self.unet_config["params"]["attention_resolutions"] = unet_attention_resolutions
        self.unet_config["params"]["num_res_blocks"] = unet_num_res_blocks
        self.unet_config["params"]["channel_mult"] = unet_channel_mult
        self.unet_config["params"]["num_head_channels"] = unet_num_head_channels
        self.unet_config["params"]["use_spatial_transformer"] = unet_use_spatial_transformer
        self.unet_config["params"]["transformer_depth"] = unet_transformer_depth

        self.first_stage_config = {}
        self.first_stage_config["target"] = ".latent_encoder.autoencoder.AudioLDMAutoencoderKL"
        self.first_stage_config["params"] = {}
        self.first_stage_config["params"]["sampling_rate"] = autoencoder_sampling_rate
        self.first_stage_config["params"]["batchsize"] = autoencoder_batchsize
        self.first_stage_config["params"]["image_key"] = autoencoder_image_key
        self.first_stage_config["params"]["subband"] = autoencoder_subband
        self.first_stage_config["params"]["embed_dim"] = autoencoder_embed_dim
        self.first_stage_config["params"]["time_shuffle"] = autoencoder_time_shuffle

        self.first_stage_config["params"]["ddconfig"] = {}
        self.first_stage_config["params"]["ddconfig"]["double_z"] = ddconfig_double_z
        self.first_stage_config["params"]["ddconfig"]["mel_bins"] = ddconfig_mel_bins
        self.first_stage_config["params"]["ddconfig"]["z_channels"] = ddconfig_z_channels
        self.first_stage_config["params"]["ddconfig"]["resolution"] = ddconfig_resolution
        self.first_stage_config["params"]["ddconfig"]["downsample_time"] = ddconfig_downsample_time
        self.first_stage_config["params"]["ddconfig"]["in_channels"] = ddconfig_in_channels
        self.first_stage_config["params"]["ddconfig"]["out_ch"] = ddconfig_out_ch
        self.first_stage_config["params"]["ddconfig"]["ch"] = ddconfig_ch
        self.first_stage_config["params"]["ddconfig"]["ch_mult"] = ddconfig_ch_mult
        self.first_stage_config["params"]["ddconfig"]["num_res_blocks"] = ddconfig_num_res_blocks
        self.first_stage_config["params"]["ddconfig"]["attn_resolutions"] = ddconfig_attn_resolutions
        self.first_stage_config["params"]["ddconfig"]["dropout"] = ddconfig_dropout

        self.cond_stage_config = {}
        self.cond_stage_config["crossattn_audiomae_generated"] = {}
        self.cond_stage_config["crossattn_audiomae_generated"]["cond_stage_key"] = "all"
        self.cond_stage_config["crossattn_audiomae_generated"]["conditioning_key"] = "crossattn"
        self.cond_stage_config["crossattn_audiomae_generated"][
            "target"
        ] = ".encoders.sequence2audiomae_encoder.SequenceGenAudioMAECond"  # gpt2
        self.cond_stage_config["crossattn_audiomae_generated"]["params"] = {}
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "always_output_audiomae_gt"
        ] = sequence2audiomae_always_output_audiomae_gt
        self.cond_stage_config["crossattn_audiomae_generated"]["params"]["learnable"] = sequence2audiomae_learnable
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "use_gt_mae_output"
        ] = sequence2audiomae_use_gt_mae_output
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "use_gt_mae_prob"
        ] = sequence2audiomae_use_gt_mae_prob
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "base_learning_rate"
        ] = sequence2audiomae_base_learning_rate
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "sequence_gen_length"
        ] = sequence2audiomae_sequence_gen_length
        self.cond_stage_config["crossattn_audiomae_generated"]["params"]["use_warmup"] = sequence2audiomae_use_warmup
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "sequence_input_key"
        ] = sequence2audiomae_sequence_input_key
        self.cond_stage_config["crossattn_audiomae_generated"]["params"][
            "sequence_input_embed_dim"
        ] = sequence2audiomae_sequence_input_embed_dim
        self.cond_stage_config["crossattn_audiomae_generated"]["params"]["batchsize"] = sequence2audiomae_batchsize

        if "speech" not in model_name:
            self.cond_stage_config["crossattn_flan_t5"] = {}
            self.cond_stage_config["crossattn_flan_t5"]["cond_stage_key"] = "text"
            self.cond_stage_config["crossattn_flan_t5"]["conditioning_key"] = "crossattn"
            self.cond_stage_config["crossattn_flan_t5"]["target"] = ".encoders.flant5_encoder.FlanT5HiddenState"

        if sequence2audiomae_cond_stage_configs is None:
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"] = {}
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "film_clap_cond1"
            ] = {}
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "cond_stage_key"
            ] = "text"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "conditioning_key"
            ] = "film"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "target"
            ] = ".encoders.clap_encoder.CLAPAudioEmbeddingClassifierFreev2"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "params"
            ] = {}
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "params"
            ]["sampling_rate"] = 48000
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "params"
            ]["embed_mode"] = "text"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["film_clap_cond1"][
                "params"
            ]["amodel"] = "HTSAT-base"

            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_flan_t5"
            ] = {}
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["crossattn_flan_t5"][
                "cond_stage_key"
            ] = "text"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["crossattn_flan_t5"][
                "conditioning_key"
            ] = "crossattn"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"]["crossattn_flan_t5"][
                "target"
            ] = ".encoders.flant5_encoder.FlanT5HiddenState"

            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ] = {}
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["cond_stage_key"] = "ta_kaldi_fbank"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["conditioning_key"] = "crossattn"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["target"] = ".encoders.audiomae_encoder.AudioMAEConditionCTPoolRand"
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"] = {}
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["regularization"] = False
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["no_audiomae_mask"] = True
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["time_pooling_factors"] = [8]
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["freq_pooling_factors"] = [8]
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["eval_time_pooling"] = 8
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["eval_freq_pooling"] = 8
            self.cond_stage_config["crossattn_audiomae_generated"]["params"]["cond_stage_config"][
                "crossattn_audiomae_pooled"
            ]["params"]["mask_ratio"] = 0
        else:
            self.cond_stage_config["crossattn_audiomae_generated"]["params"][
                "cond_stage_config"
            ] = sequence2audiomae_cond_stage_configs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
