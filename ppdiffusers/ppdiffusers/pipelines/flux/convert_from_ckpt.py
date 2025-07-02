# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" Conversion script for the Stable Diffusion checkpoints."""


from typing import Dict, Optional, Union

import paddle

from ppdiffusers.transformers import (
    AutoConfig,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from ...models import AutoencoderKL, FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_omegaconf_available, is_paddlenlp_available, logging, smart_load
from ...utils.import_utils import BACKENDS_MAPPING
from ..pipeline_utils import DiffusionPipeline

# from .safety_checker import StableDiffusionSafetyChecker
# from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

# if is_accelerate_available():
#     from accelerate import init_empty_weights
#     from ..models.modeling_utils import load_model_dict_into_meta
LDM_VAE_KEYS = ["first_stage_model.", "vae."]
DIFFUSERS_TO_LDM_MAPPING = {
    "unet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "controlnet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "controlnet_cond_embedding.conv_in.weight": "input_hint_block.0.weight",
            "controlnet_cond_embedding.conv_in.bias": "input_hint_block.0.bias",
            "controlnet_cond_embedding.conv_out.weight": "input_hint_block.14.weight",
            "controlnet_cond_embedding.conv_out.bias": "input_hint_block.14.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "vae": {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    },
    "openclip": {
        "layers": {
            "text_model.embeddings.position_embedding.weight": "positional_embedding",
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.final_layer_norm.weight": "ln_final.weight",
            "text_model.final_layer_norm.bias": "ln_final.bias",
            "text_projection.weight": "text_projection",
        },
        "transformer": {
            "text_model.encoder.layers.": "resblocks.",
            "layer_norm1": "ln_1",
            "layer_norm2": "ln_2",
            ".fc1.": ".c_fc.",
            ".fc2.": ".c_proj.",
            ".self_attn": ".attn",
            "transformer.text_model.final_layer_norm.": "ln_final.",
            "transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "transformer.text_model.embeddings.position_embedding.weight": "positional_embedding",
        },
    },
}
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from ...models.modeling_utils import ContextManagers, faster_set_state_dict

if is_paddlenlp_available():
    try:
        from paddlenlp.transformers.model_utils import no_init_weights
    except ImportError:
        from ...utils.paddle_utils import no_init_weights


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = (
            ldm_key.replace(mapping["old"], mapping["new"])
            .replace("norm.weight", "group_norm.weight")
            .replace("norm.bias", "group_norm.bias")
            .replace("q.weight", "to_q.weight")
            .replace("q.bias", "to_q.bias")
            .replace("k.weight", "to_k.weight")
            .replace("k.bias", "to_k.bias")
            .replace("v.weight", "to_v.weight")
            .replace("v.bias", "to_v.bias")
            .replace("proj_out.weight", "to_out.0.weight")
            .replace("proj_out.bias", "to_out.0.bias")
        )
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)

        # proj_attn.weight has to be converted from conv 1D to linear
        shape = new_checkpoint[diffusers_key].shape

        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    # remove the LDM_VAE_KEY prefix from the ldm checkpoint keys so that it is easier to map them to diffusers keys
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = ""
    for ldm_vae_key in LDM_VAE_KEYS:
        if any(k.startswith(ldm_vae_key) for k in keys):
            vae_key = ldm_vae_key

    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["vae"]
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(config["down_block_types"])
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.bias"
            )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint


def convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())

    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "double_blocks." in k))[-1] + 1  # noqa: C401
    num_single_layers = (
        list(set(int(k.split(".", 2)[1]) for k in checkpoint if "single_blocks." in k))[-1] + 1
    )  # noqa: C401
    mlp_ratio = 4.0
    inner_dim = 3072

    # in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
    # while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
    def swap_scale_shift(weight):
        weight = paddle.to_tensor(weight)
        shift, scale = weight.chunk(chunks=2, axis=0)
        new_weight = paddle.concat([scale, shift], axis=0)
        return new_weight

    # time_text_embed.timestep_embedder <-  time_in
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("time_in.in_layer.bias")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("time_in.out_layer.bias")

    # time_text_embed.text_embedder <- vector_in
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = checkpoint.pop("vector_in.in_layer.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = checkpoint.pop("vector_in.in_layer.bias")
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = checkpoint.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = checkpoint.pop("vector_in.out_layer.bias")

    # guidance
    has_guidance = any("guidance" in k for k in checkpoint)
    if has_guidance:
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = checkpoint.pop(
            "guidance_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = checkpoint.pop(
            "guidance_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = checkpoint.pop(
            "guidance_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = checkpoint.pop(
            "guidance_in.out_layer.bias"
        )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = checkpoint.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = checkpoint.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms.
        # norm1
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        # norm1_context
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )
        # Q, K, V
        sample_q, sample_k, sample_v = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.weight")), chunks=3, axis=0
        )
        context_q, context_k, context_v = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.weight")), chunks=3, axis=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.bias")), chunks=3, axis=0
        )
        context_q_bias, context_k_bias, context_v_bias = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.bias")), chunks=3, axis=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = sample_q
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = sample_q_bias
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = sample_k
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = sample_k_bias
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = sample_v
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = sample_v_bias
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = context_q
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = context_q_bias
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = context_k
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = context_k_bias
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = context_v
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = context_v_bias
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )
        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.0.bias")
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.weight")
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.bias")
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        # q, k, v, mlp = torch.split(checkpoint.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q, k, v, mlp = paddle.split(
            paddle.to_tensor(checkpoint.pop(f"single_blocks.{i}.linear1.weight")), num_or_sections=split_size, axis=0
        )
        q_bias, k_bias, v_bias, mlp_bias = paddle.split(
            paddle.to_tensor(checkpoint.pop(f"single_blocks.{i}.linear1.bias")), num_or_sections=split_size, axis=0
        )
        # q_bias, k_bias, v_bias, mlp_bias = torch.split(
        #     checkpoint.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        # )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = q
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = q_bias
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = k
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = k_bias
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = v
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = v_bias
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = mlp
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = mlp_bias
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = checkpoint.pop(f"single_blocks.{i}.linear2.weight")
        converted_state_dict[f"{block_prefix}proj_out.bias"] = checkpoint.pop(f"single_blocks.{i}.linear2.bias")

    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias")
    )

    return converted_state_dict


def convert_diffusers_vae_unet_to_ppdiffusers(vae_or_unet, diffusers_vae_unet_checkpoint):
    import paddle.nn as nn

    need_transpose = []
    for k, v in vae_or_unet.named_sublayers(include_self=True):
        if isinstance(v, nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = {}
    for k in list(diffusers_vae_unet_checkpoint.keys()):
        v = diffusers_vae_unet_checkpoint.pop(k)
        if k not in need_transpose:
            new_vae_or_unet[k] = v
        else:
            new_vae_or_unet[k] = v.T
    return new_vae_or_unet


def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False, text_encoder=None):
    if text_encoder is None:
        config_name = "openai/clip-vit-large-patch14"
        try:
            config = CLIPTextConfig.from_pretrained(config_name, local_files_only=local_files_only)
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
            )
        init_contexts = []
        init_contexts.append(paddle.dtype_guard(paddle.float32))
        init_contexts.append(no_init_weights(_enable=True))
        if hasattr(paddle, "LazyGuard"):
            init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            text_model = CLIPTextModel(config)
    else:
        text_model = text_encoder

    keys = list(checkpoint.keys())

    text_model_dict = {}

    remove_prefixes = [
        "cond_stage_model.transformer",
        "conditioner.embedders.0.transformer",
        "text_encoders.clip_l.transformer",
    ]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                text_model_dict[key[len(prefix + ".") :]] = checkpoint[key]

    if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
        text_model_dict.pop("text_model.embeddings.position_ids", None)

    faster_set_state_dict(text_model, convert_diffusers_vae_unet_to_ppdiffusers(text_model, text_model_dict))

    return text_model


def convert_sd3_t5_checkpoint_to_diffusers(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = ["text_encoders.t5xxl.transformer."]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


def create_diffusers_t5_model_from_checkpoint(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
):
    if config:
        config = {"pretrained_model_name_or_path": config}
    config = AutoConfig.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2")
    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        model = T5EncoderModel(config)

    diffusers_format_checkpoint = convert_sd3_t5_checkpoint_to_diffusers(checkpoint)

    faster_set_state_dict(model, convert_diffusers_vae_unet_to_ppdiffusers(model, diffusers_format_checkpoint))

    use_keep_in_fp32_modules = (T5EncoderModel._keep_in_fp32_modules is not None) and (torch_dtype == paddle.float16)
    if use_keep_in_fp32_modules:
        keep_in_fp32_modules = model._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []

    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                # param = param.to(torch.float32) does not work here as only in the local scope.
                param.data = param.data.to(paddle.float32)

    return model


def download_from_original_flux_ckpt(
    checkpoint_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
    original_config_file: str = None,
    image_size: Optional[int] = None,
    prediction_type: str = None,
    model_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: Optional[int] = None,
    upcast_attention: Optional[bool] = None,
    device: str = None,
    from_safetensors: bool = False,
    stable_unclip: Optional[str] = None,
    stable_unclip_prior: Optional[str] = None,
    clip_stats_path: Optional[str] = None,
    controlnet: Optional[bool] = None,
    adapter: Optional[bool] = None,
    load_safety_checker: bool = True,
    pipeline_class: DiffusionPipeline = None,
    local_files_only=False,
    vae_path=None,
    vae=None,
    text_encoder=None,
    tokenizer=None,
    config_files=None,
    paddle_dtype=None,
    **kwargs,
) -> DiffusionPipeline:
    """
    Load a Stable Diffusion pipeline object from a CompVis-style `.ckpt`/`.safetensors` file and (ideally) a `.yaml`
    config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path_or_dict (`str` or `dict`): Path to `.ckpt` file, or the state dict.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            inferred by looking for a key that only exists in SD2.0 models.
        image_size (`int`, *optional*, defaults to 512):
            The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2
            Base. Use 768 for Stable Diffusion v2.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion v1.X and Stable
            Diffusion v2 Base. Use `'v_prediction'` for Stable Diffusion v2.
        num_in_channels (`int`, *optional*, defaults to None):
            The number of input channels. If `None`, it will be automatically inferred.
        scheduler_type (`str`, *optional*, defaults to 'pndm'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        model_type (`str`, *optional*, defaults to `None`):
            The pipeline type. `None` to automatically infer, or one of `["FrozenOpenCLIPEmbedder",
            "FrozenCLIPEmbedder", "PaintByExample"]`.
        is_img2img (`bool`, *optional*, defaults to `False`):
            Whether the model should be loaded as an img2img pipeline.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        upcast_attention (`bool`, *optional*, defaults to `None`):
            Whether the attention computation should always be upcasted. This is necessary when running stable
            diffusion 2.1.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of Paddle.
        load_safety_checker (`bool`, *optional*, defaults to `True`):
            Whether to load the safety checker or not. Defaults to `True`.
        pipeline_class (`str`, *optional*, defaults to `None`):
            The pipeline class to use. Pass `None` to determine automatically.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        vae (`AutoencoderKL`, *optional*, defaults to `None`):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
            this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
        text_encoder (`CLIPTextModel`, *optional*, defaults to `None`):
            An instance of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)
            to use, specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
            variant. If this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
        tokenizer (`CLIPTokenizer`, *optional*, defaults to `None`):
            An instance of
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
            to use. If this parameter is `None`, the function will load a new instance of [CLIPTokenizer] by itself, if
            needed.
        config_files (`Dict[str, str]`, *optional*, defaults to `None`):
            A dictionary mapping from config file names to their contents. If this parameter is `None`, the function
            will load the config files by itself, if needed. Valid keys are:
                - `v1`: Config file for Stable Diffusion v1
                - `v2`: Config file for Stable Diffusion v2
                - `xl`: Config file for Stable Diffusion XL
                - `xl_refiner`: Config file for Stable Diffusion XL Refiner
        return: A StableDiffusionPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    # import pipelines here to avoid circular import error when using from_single_file method
    from ppdiffusers import FluxPipeline

    if not is_omegaconf_available():
        raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

    if isinstance(checkpoint_path_or_dict, str):
        checkpoint = smart_load(checkpoint_path_or_dict, return_numpy=True, return_global_step=True)

    elif isinstance(checkpoint_path_or_dict, dict):
        checkpoint = checkpoint_path_or_dict

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    global_step = int(checkpoint.pop("global_step", -1))

    if global_step == -1:
        print("global_step key not found in model")

    # must cast them to float32
    newcheckpoint = {}
    for k, v in checkpoint.items():
        try:
            if "int" in str(v.dtype):
                continue
        except Exception:
            continue
        newcheckpoint[k] = v.astype("float32")
    checkpoint = newcheckpoint

    # 此处先针对文生图的FLUX 没有针对其余功能的FLUX
    if (model_type is None) and (pipeline_class == FluxPipeline):
        model_type = "Flux"
        if image_size is None:
            image_size = 1024

    if model_type in ["Flux"]:
        scheduler_dict = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 3.0,
            "use_dynamic_shifting": True,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_dict)
    else:
        pass

    converted_flux_checkpoint = convert_flux_transformer_checkpoint_to_diffusers(checkpoint)

    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        flux_transformer = FluxTransformer2DModel(guidance_embeds=True)

    faster_set_state_dict(
        flux_transformer, convert_diffusers_vae_unet_to_ppdiffusers(flux_transformer, converted_flux_checkpoint)
    )

    # Convert the VAE model.
    if vae_path is None and vae is None:
        vae_config = AutoencoderKL.load_config("black-forest-labs/FLUX.1-dev", subfolder="vae")
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
        vae_path
        init_contexts = []
        init_contexts.append(paddle.dtype_guard(paddle.float32))
        init_contexts.append(no_init_weights(_enable=True))
        if hasattr(paddle, "LazyGuard"):
            init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            vae = AutoencoderKL(**vae_config)

        faster_set_state_dict(vae, convert_diffusers_vae_unet_to_ppdiffusers(vae, converted_vae_checkpoint))

    elif vae is None:
        vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=local_files_only)

    if model_type in ["Flux"]:

        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
            )
        text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=local_files_only)

        # try:
        tokenizer_2 = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", local_files_only=local_files_only)
        # except Exception:
        #     raise ValueError(
        #         f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' with `pad_token` set to '!'."
        #     )

        text_encoder_2 = create_diffusers_t5_model_from_checkpoint(
            pipeline_class,
            checkpoint,
            subfolder="",
            config=None,
            torch_dtype=None,
            local_files_only=local_files_only,
        )
        # text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev",subfolder="text_encoder_2" , paddle_dtype="float32" )

        pipe = pipeline_class(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=flux_transformer,
            scheduler=scheduler,
            # force_zeros_for_empty_prompt=True,
        )

    if paddle_dtype is not None:
        pipe.to(paddle_dtype=paddle_dtype)

    return pipe
