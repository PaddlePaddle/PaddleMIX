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
import os

import paddle
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description="Convert model weight to pd.")
    parser.add_argument("--model_name", type=str, help="Your src model path")
    args = parser.parse_args()
    # 模型路径在此修改
    model_name = args.model_name
    # [
    #     "data/data268105/animagine-xl-3.1.safetensors",
    #     "data/data268105/animagine-xl-3.1.safetensors"
    # ][-1]

    model_dir = os.path.splitext(os.path.basename(model_name))[0].title()
    os.makedirs(model_dir, exist_ok=True)
    os.system(f"cp -r convert_single_weight_pt_2_pd/basemodel/* {model_dir}")

    with open("convert_single_weight_pt_2_pd/text_pt.txt", "r") as f:
        key_list_text = f.readlines()
    with open("convert_single_weight_pt_2_pd/text_2_pt.txt", "r") as f:
        key_list_text_2 = f.readlines()
    with open("convert_single_weight_pt_2_pd/vae_pt.txt", "r") as f:
        key_list_vae = f.readlines()
    with open("convert_single_weight_pt_2_pd/unet_pt.txt", "r", encoding="utf-8") as f:
        unet_key_pt = f.readlines()
    with open("convert_single_weight_pt_2_pd/unet_pd.txt", "r", encoding="utf-8") as f:
        unet_key_pd = f.readlines()

    f2 = safe_open(model_name, framework="np")

    # text
    state_dict_text = {}
    for key in key_list_text:
        try:
            k = key.strip()
            v = f2.get_tensor(k)
            if len(v.shape) == 2 and not ("position_embedding" in k or "token_embedding" in k):
                state_dict_text[k.replace("conditioner.embedders.0.transformer.", "")] = paddle.to_tensor(
                    v, dtype=v.dtype
                ).t()
            else:
                state_dict_text[k.replace("conditioner.embedders.0.transformer.", "")] = paddle.to_tensor(
                    v, dtype=v.dtype
                )
        except Exception as e:
            print("错误原因：", e)
    paddle.save(state_dict_text, model_dir + "/text_encoder/model_state.pdparams")
    paddle.device.cuda.empty_cache()
    print("text_encode_1 Done.")

    # text_2
    tect_2_k_map = [
        ["text_model.embeddings.position_embedding.weight", "conditioner.embedders.1.model.positional_embedding"],
        ["text_model.embeddings.token_embedding.weight", "conditioner.embedders.1.model.token_embedding.weight"],
        ["text_projection.weight", "conditioner.embedders.1.model.text_projection"],
        ["text_model.encoder.layers", "conditioner.embedders.1.model.transformer.resblocks"],
        ["text_model", "conditioner.embedders.1.model"],
        ["self_attn", "attn"],
        ["fc1", "c_fc"],
        ["fc2", "c_proj"],
        ["final_layer_norm", "ln_final"],
        ["layer_norm", "ln_"],
    ]
    qkv_map = [["q_proj.", "k_proj.", "v_proj."], "in_proj_"]

    state_dict_text_2 = {}
    for key in key_list_text_2:
        try:
            k = key.strip()
            v = f2.get_tensor(k)
            for k_map in tect_2_k_map:
                k = k.replace(k_map[1], k_map[0])
            if "in_proj" in k and "bias" in k:
                q_pd, k_pd, v_pd = paddle.to_tensor(v, dtype=v.dtype).chunk(3, axis=0)
                state_dict_text_2[k.replace(qkv_map[1], qkv_map[0][0])] = q_pd
                state_dict_text_2[k.replace(qkv_map[1], qkv_map[0][1])] = k_pd
                state_dict_text_2[k.replace(qkv_map[1], qkv_map[0][2])] = v_pd
                continue
            if "in_proj" in k and "weight" in k:
                q_pd, k_pd, v_pd = paddle.to_tensor(v, dtype=v.dtype).t().chunk(3, axis=1)
                state_dict_text_2[k.replace(qkv_map[1], qkv_map[0][0])] = q_pd  # .t()
                state_dict_text_2[k.replace(qkv_map[1], qkv_map[0][1])] = k_pd  # .t()
                state_dict_text_2[k.replace(qkv_map[1], qkv_map[0][2])] = v_pd  # .t()
                continue
            if len(v.shape) == 2 and not ("position_embedding" in k or "token_embedding" in k):
                state_dict_text_2[k] = paddle.to_tensor(v, dtype=v.dtype).t()
            else:
                state_dict_text_2[k] = paddle.to_tensor(v, dtype=v.dtype)
        except Exception as e:
            print("错误原因：", e)

    paddle.save(state_dict_text_2, model_dir + "/text_encoder_2/model_state.pdparams")
    paddle.device.cuda.empty_cache()
    print("text_encode_2 Done.")

    # vae
    vae_k_map = [
        ["", "first_stage_model."],
        ["conv_norm_out", "norm_out"],
        [".to_k.", ".k."],
        [".to_out.0.", ".proj_out."],
        [".to_q.", ".q."],
        [".to_v.", ".v."],
        ["mid.attn_1.group_norm", "mid.attn_1.norm"],
        ["mid_block.attentions.0", "mid.attn_1"],
        ["mid_block.resnets.0", "mid.block_1"],
        ["mid_block.resnets.1", "mid.block_2"],
        ["up_blocks.0.resnets", "up.3.block"],
        ["up_blocks.0.upsamplers.0", "up.3.upsample"],
        ["up_blocks.1.resnets", "up.2.block"],
        ["up_blocks.1.upsamplers.0", "up.2.upsample"],
        ["up_blocks.2.resnets", "up.1.block"],
        ["up_blocks.2.upsamplers.0", "up.1.upsample"],
        ["up_blocks.3.resnets", "up.0.block"],
        ["down_blocks.0.resnets", "down.0.block"],
        ["down_blocks.0.downsamplers.0", "down.0.downsample"],
        ["down_blocks.1.resnets", "down.1.block"],
        ["down_blocks.1.downsamplers.0", "down.1.downsample"],
        ["down_blocks.2.resnets", "down.2.block"],
        ["down_blocks.2.downsamplers.0", "down.2.downsample"],
        ["down_blocks.3.resnets", "down.3.block"],
        ["conv_shortcut", "nin_shortcut"],
    ]
    state_dict_vae = {}
    for key in key_list_vae:
        try:
            k = key.strip()
            v = f2.get_tensor(k)
            for k_map in vae_k_map:
                k = k.replace(k_map[1], k_map[0])
            if len(v.shape) == 2:
                state_dict_vae[k] = paddle.to_tensor(v, v.dtype).t()
            elif "att" in k and len(v.shape) == 4:
                state_dict_vae[k] = paddle.to_tensor(v.squeeze(), v.dtype).t()
            else:
                state_dict_vae[k] = paddle.to_tensor(v, v.dtype)
        except Exception as e:
            print("错误原因：", e)
    paddle.save(state_dict_vae, model_dir + "/vae/model_state.pdparams")
    paddle.device.cuda.empty_cache()
    print("Vae Done.")

    # unet
    state_dict_unet = {}
    for k_pt, k_pd in zip(unet_key_pt, unet_key_pd):
        try:
            k_pt = k_pt.strip()
            k_pd = k_pd.strip()
            v = f2.get_tensor(k_pt)
            if len(v.shape) == 2:
                state_dict_unet[k_pd] = paddle.to_tensor(v, v.dtype).t()
            else:
                state_dict_unet[k_pd] = paddle.to_tensor(v, v.dtype)
        except Exception as e:
            print(e)
    paddle.save(state_dict_unet, model_dir + "/unet/model_state.pdparams")
    paddle.device.cuda.empty_cache()
    print("Unet Done.")


if __name__ == "__main__":
    main()
