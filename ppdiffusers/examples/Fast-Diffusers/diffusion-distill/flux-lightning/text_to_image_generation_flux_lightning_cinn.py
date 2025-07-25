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

import argparse
import datetime
import os

os.environ["USE_PEFT_BACKEND"] = "True"
import paddle

from ppdiffusers import FluxPipeline

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--path_to_lora",
    type=str,
    required=True,
    help="Path to paddle_lora_weights.safetensors",
)
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    default="a beautiful girl",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    required=False,
    default=3.5,
)
parser.add_argument(
    "--height",
    type=int,
    required=False,
    default=1024,
)
parser.add_argument(
    "--width",
    type=int,
    required=False,
    default=1024,
)
parser.add_argument(
    "--lora_scale",
    type=float,
    required=False,
    default=0.25,
)
parser.add_argument(
    "--step",
    type=int,
    required=False,
    default=4,
)
parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=False,
    default="./",
)
parser.add_argument(
    "--inference_optimize",
    action="store_true",
    help="Whether or not to use cinn.",
)

args = parser.parse_args()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", map_location="cpu", paddle_dtype=paddle.bfloat16)
pipe.load_lora_weights(args.path_to_lora)

if args.inference_optimize:
    pipe.transformer.jit_transformer_blocks = paddle.incubate.jit.inference(
        pipe.transformer.jit_transformer_blocks,
        save_model_dir="./tmp/flux",
        enable_new_ir=True,
        switch_ir_optim=True,
        cache_static_model=False,
        exp_enable_use_cutlass=False,
        delete_pass_lists=[
            "add_norm_fuse_pass",
            "add_shadow_output_after_dead_parameter_pass",
            "delete_quant_dequant_linear_op_pass",
            "delete_weight_dequant_linear_op_pass",
            "map_op_to_another_pass",
            "identity_op_clean_pass",
            "silu_fuse_pass",
            "conv2d_bn_fuse_pass",
            "conv2d_add_act_fuse_pass",
            "conv2d_add_fuse_pass",
            "embedding_eltwise_layernorm_fuse_pass",
            "fused_rotary_position_embedding_pass",
            "fused_flash_attn_pass",
            "multihead_matmul_fuse_pass",
            "matmul_add_act_fuse_pass",
            "fc_elementwise_layernorm_fuse_pass",
            "group_norm_silu_fuse_pass",
            "matmul_scale_fuse_pass",
            "matmul_transpose_fuse_pass",
            "transpose_flatten_concat_fuse_pass",
            "remove_redundant_transpose_pass",
            "horizontal_fuse_pass",
            "common_subexpression_elimination_pass",
        ],
    )
    pipe.transformer.jit_single_transformer_blocks = paddle.incubate.jit.inference(
        pipe.transformer.jit_single_transformer_blocks,
        save_model_dir="./tmp/flux",
        enable_new_ir=True,
        switch_ir_optim=True,
        cache_static_model=False,
        exp_enable_use_cutlass=False,
        delete_pass_lists=[
            "add_norm_fuse_pass",
            "add_shadow_output_after_dead_parameter_pass",
            "delete_quant_dequant_linear_op_pass",
            "delete_weight_dequant_linear_op_pass",
            "map_op_to_another_pass",
            "identity_op_clean_pass",
            "silu_fuse_pass",
            "conv2d_bn_fuse_pass",
            "conv2d_add_act_fuse_pass",
            "conv2d_add_fuse_pass",
            "embedding_eltwise_layernorm_fuse_pass",
            "fused_rotary_position_embedding_pass",
            "fused_flash_attn_pass",
            "multihead_matmul_fuse_pass",
            "matmul_add_act_fuse_pass",
            "fc_elementwise_layernorm_fuse_pass",
            "group_norm_silu_fuse_pass",
            "matmul_scale_fuse_pass",
            "matmul_transpose_fuse_pass",
            "transpose_flatten_concat_fuse_pass",
            "remove_redundant_transpose_pass",
            "horizontal_fuse_pass",
            "common_subexpression_elimination_pass",
        ],
    )
    pipe.transformer.forward = pipe.transformer.jit_forward

# warmup
for i in range(3):
    with paddle.no_grad():
        result_image = pipe(
            prompt=args.prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_inference_steps=args.step,
            guidance_scale=args.guidance_scale,
            generator=paddle.Generator().manual_seed(args.seed),
            joint_attention_kwargs={"scale": args.lora_scale},
        ).images[0]

paddle.device.synchronize()

repeat_times = 3
sumtime = 0.0

for i in range(repeat_times):
    starttime = datetime.datetime.now()
    with paddle.no_grad():
        result_image = pipe(
            prompt=args.prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_inference_steps=args.step,
            guidance_scale=args.guidance_scale,
            generator=paddle.Generator().manual_seed(args.seed),
            joint_attention_kwargs={"scale": args.lora_scale},
        ).images[0]
        endtime = datetime.datetime.now()
        duringtime = endtime - starttime
        duringtime = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        sumtime += duringtime

print("FLUX average end-to-end time : ", sumtime / repeat_times, "ms")

result_image.save(os.path.join(args.output_dir, "test_flux_lightning_cinn.png"))
