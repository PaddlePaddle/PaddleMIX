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

import argparse

import paddle

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiTPipeline,
    DPMSolverMultistepScheduler,
    Transformer2DModel,
)

# num_layers(depth), hidden_size, patch_size, num_heads
arch_settings = {
    "DiT_XL_2": [28, 1152, 2, 16],
    "DiT_XL_4": [28, 1152, 4, 16],
    "DiT_XL_8": [28, 1152, 8, 16],
    "DiT_L_2": [24, 1024, 2, 16],
    "DiT_L_4": [24, 1024, 4, 16],
    "DiT_L_8": [24, 1024, 8, 16],
    "DiT_B_2": [12, 768, 2, 12],
    "DiT_B_4": [12, 768, 4, 12],
    "DiT_B_8": [12, 768, 8, 12],
    "DiT_S_2": [12, 384, 2, 6],
    "DiT_S_4": [12, 384, 2, 6],
    "DiT_S_8": [12, 384, 2, 6],
}


def main(args):
    num_layers, hidden_size, patch_size, num_heads = arch_settings[args.model_name]

    state_dict_prefix = paddle.load(args.model_weights)
    state_dict = {k.replace("transformer.", ""): v for k, v in state_dict_prefix.items()}
    del state_dict_prefix

    state_dict["pos_embed.proj.weight"] = state_dict["x_embedder.proj.weight"]
    state_dict["pos_embed.proj.bias"] = state_dict["x_embedder.proj.bias"]
    state_dict.pop("x_embedder.proj.weight")
    state_dict.pop("x_embedder.proj.bias")

    for depth in range(num_layers):
        state_dict[f"transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_1.weight"] = state_dict[
            "t_embedder.mlp.0.weight"
        ]
        state_dict[f"transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_1.bias"] = state_dict[
            "t_embedder.mlp.0.bias"
        ]
        state_dict[f"transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_2.weight"] = state_dict[
            "t_embedder.mlp.2.weight"
        ]
        state_dict[f"transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_2.bias"] = state_dict[
            "t_embedder.mlp.2.bias"
        ]
        state_dict[f"transformer_blocks.{depth}.norm1.emb.class_embedder.embedding_table.weight"] = state_dict[
            "y_embedder.embedding_table.weight"
        ]

        state_dict[f"transformer_blocks.{depth}.norm1.linear.weight"] = state_dict[
            f"blocks.{depth}.adaLN_modulation.1.weight"
        ]
        state_dict[f"transformer_blocks.{depth}.norm1.linear.bias"] = state_dict[
            f"blocks.{depth}.adaLN_modulation.1.bias"
        ]

        q, k, v = paddle.chunk(state_dict[f"blocks.{depth}.attn.qkv.weight"], 3, axis=1)  # torch axis=0
        q_bias, k_bias, v_bias = paddle.chunk(state_dict[f"blocks.{depth}.attn.qkv.bias"], 3, axis=0)

        state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        state_dict[f"transformer_blocks.{depth}.attn1.to_q.bias"] = q_bias
        state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        state_dict[f"transformer_blocks.{depth}.attn1.to_k.bias"] = k_bias
        state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        state_dict[f"transformer_blocks.{depth}.attn1.to_v.bias"] = v_bias

        state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict[
            f"blocks.{depth}.attn.proj.weight"
        ]
        state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict[f"blocks.{depth}.attn.proj.bias"]

        state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.weight"] = state_dict[f"blocks.{depth}.mlp.fc1.weight"]
        state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.bias"] = state_dict[f"blocks.{depth}.mlp.fc1.bias"]
        state_dict[f"transformer_blocks.{depth}.ff.net.2.weight"] = state_dict[f"blocks.{depth}.mlp.fc2.weight"]
        state_dict[f"transformer_blocks.{depth}.ff.net.2.bias"] = state_dict[f"blocks.{depth}.mlp.fc2.bias"]

        state_dict.pop(f"blocks.{depth}.attn.qkv.weight")
        state_dict.pop(f"blocks.{depth}.attn.qkv.bias")
        state_dict.pop(f"blocks.{depth}.attn.proj.weight")
        state_dict.pop(f"blocks.{depth}.attn.proj.bias")
        state_dict.pop(f"blocks.{depth}.mlp.fc1.weight")
        state_dict.pop(f"blocks.{depth}.mlp.fc1.bias")
        state_dict.pop(f"blocks.{depth}.mlp.fc2.weight")
        state_dict.pop(f"blocks.{depth}.mlp.fc2.bias")
        state_dict.pop(f"blocks.{depth}.adaLN_modulation.1.weight")
        state_dict.pop(f"blocks.{depth}.adaLN_modulation.1.bias")

    state_dict.pop("t_embedder.mlp.0.weight")
    state_dict.pop("t_embedder.mlp.0.bias")
    state_dict.pop("t_embedder.mlp.2.weight")
    state_dict.pop("t_embedder.mlp.2.bias")
    state_dict.pop("y_embedder.embedding_table.weight")

    state_dict["proj_out_1.weight"] = state_dict["final_layer.adaLN_modulation.1.weight"]
    state_dict["proj_out_1.bias"] = state_dict["final_layer.adaLN_modulation.1.bias"]
    state_dict["proj_out_2.weight"] = state_dict["final_layer.linear.weight"]
    state_dict["proj_out_2.bias"] = state_dict["final_layer.linear.bias"]

    state_dict.pop("final_layer.linear.weight")
    state_dict.pop("final_layer.linear.bias")
    state_dict.pop("final_layer.adaLN_modulation.1.weight")
    state_dict.pop("final_layer.adaLN_modulation.1.bias")

    # default DiT XL/2
    transformer = Transformer2DModel(
        sample_size=args.image_size // 8,
        num_layers=num_layers,  #
        attention_head_dim=hidden_size // num_heads,  #
        in_channels=4,
        out_channels=8,
        patch_size=patch_size,  #
        attention_bias=True,
        num_attention_heads=num_heads,  #
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
    )
    transformer.set_state_dict(state_dict)

    if args.scheduler == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,
        )
    else:
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

    vae = AutoencoderKL.from_pretrained(args.vae_model)

    pipeline = DiTPipeline(transformer=transformer, vae=vae, scheduler=scheduler)

    if args.save:
        pipeline.save_pretrained(args.checkpoint_path, safe_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        required=False,
        help="Image size of pretrained model, either 256 or 512.",
    )
    parser.add_argument(
        "--model_name",
        default="DiT_XL_2",
        type=str,
        required=False,
        help="DiT model name.",
    )
    parser.add_argument(
        "--vae_model",
        default="stabilityai/sd-vae-ft-mse",
        type=str,
        required=False,
        help="Path to pretrained VAE model, either stabilityai/sd-vae-ft-mse or stabilityai/sd-vae-ft-ema.",
    )
    parser.add_argument(
        "--scheduler",
        default="ddim",
        type=str,
        required=False,
        help="DiTPipeline sample scheduler",
    )
    parser.add_argument(
        "--model_weights",
        default="DiT-XL-2-256x256.pdparams",
        type=str,
        required=False,
        help="model weights path.",
    )

    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted pipeline or not."
    )
    parser.add_argument(
        "--checkpoint_path", default="DiT_XL_2_256/", type=str, required=False, help="Path to the output pipeline."
    )

    args = parser.parse_args()
    main(args)

# python tools/convert_dit_to_ppdiffusers.py --image_size 512 --model_name DiT_XL_2 --model_weights DiT-XL-2-512x512.pdparams --checkpoint_path DiT_XL_2_512
# python tools/convert_dit_to_ppdiffusers.py --image_size 256 --model_name DiT_XL_2 --model_weights DiT-XL-2-256x256.pdparams --checkpoint_path DiT_XL_2_256
# python tools/convert_dit_to_ppdiffusers.py --image_size 256 --model_name DiT_XL_2 --model_weights SiT-XL-2-256x256.pdparams --vae_model stabilityai/sd-vae-ft-ema --checkpoint_path SiT_XL_2_256

# python tools/convert_dit_to_ppdiffusers.py --image_size 512 --model_name DiT_XL_2 --model_weights DiT_XL_patch2_256_global_steps_1000.pdparams --checkpoint_path DiT_XL_2_256
# python tools/convert_dit_to_ppdiffusers.py --image_size 256 --model_name DiT_XL_2 --model_weights output_notrainer/000-DiT_XL_patch2/checkpoints/0001000_ema.pdparams --checkpoint_path DiT_XL_2_256
