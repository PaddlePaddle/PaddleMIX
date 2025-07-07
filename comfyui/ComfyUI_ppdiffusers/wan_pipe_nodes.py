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


import json
import os

import folder_paths
import numpy as np
import paddle
import torch  # for convert data
from comfy.cli_args import args
from comfy.comfy_types import FileLocator
from PIL import Image
from safetensors.torch import load_file

from ppdiffusers import WanPipeline
from ppdiffusers.models.autoencoder_kl_wan import AutoencoderKLWan
from ppdiffusers.models.modeling_utils import faster_set_state_dict
from ppdiffusers.models.transformer_wan import WanTransformer3DModel
from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from ppdiffusers.transformers import T5Tokenizer, UMT5EncoderModel
from ppdiffusers.transformers.umt5.configuration import UMT5WANConfig


def convert_wan_transformer_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}

    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    TRANSFORMER_KEYS_RENAME_DICT = {
        "time_embedding.0": "condition_embedder.time_embedder.linear_1",
        "time_embedding.2": "condition_embedder.time_embedder.linear_2",
        "text_embedding.0": "condition_embedder.text_embedder.linear_1",
        "text_embedding.2": "condition_embedder.text_embedder.linear_2",
        "time_projection.1": "condition_embedder.time_proj",
        "cross_attn": "attn2",
        "self_attn": "attn1",
        ".o.": ".to_out.0.",
        ".q.": ".to_q.",
        ".k.": ".to_k.",
        ".v.": ".to_v.",
        ".k_img.": ".add_k_proj.",
        ".v_img.": ".add_v_proj.",
        ".norm_k_img.": ".norm_added_k.",
        "head.modulation": "scale_shift_table",
        "head.head": "proj_out",
        "modulation": "scale_shift_table",
        "ffn.0": "ffn.net.0.proj",
        "ffn.2": "ffn.net.2",
        # Hack to swap the layer names
        # The original model calls the norms in following order: norm1, norm3, norm2
        # We convert it to: norm1, norm2, norm3
        "norm2": "norm__placeholder",
        "norm3": "norm2",
        "norm__placeholder": "norm3",
        # For the I2V model
        "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
        "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
        "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
        "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    }

    for key in list(checkpoint.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)

        converted_state_dict[new_key] = checkpoint.pop(key)

    return converted_state_dict


def convert_wan_vae_to_diffusers(checkpoint, **kwargs):

    converted_state_dict = {}

    # Create mappings for specific components
    middle_key_mapping = {
        # Encoder middle block
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        # Decoder middle block
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Create a mapping for attention blocks
    attention_mapping = {
        # Encoder middle attention
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        # Decoder middle attention
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Create a mapping for the head components
    head_mapping = {
        # Encoder head
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # Decoder head
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Create a mapping for the quant components
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    # Process each key in the state dict
    for key, value in checkpoint.items():
        # Handle middle block keys using the mapping
        if key in middle_key_mapping:
            new_key = middle_key_mapping[key]
            converted_state_dict[new_key] = value
        # Handle attention blocks using the mapping
        elif key in attention_mapping:
            new_key = attention_mapping[key]
            converted_state_dict[new_key] = value
        # Handle head keys using the mapping
        elif key in head_mapping:
            new_key = head_mapping[key]
            converted_state_dict[new_key] = value
        # Handle quant keys using the mapping
        elif key in quant_mapping:
            new_key = quant_mapping[key]
            converted_state_dict[new_key] = value
        # Handle encoder conv1
        elif key == "encoder.conv1.weight":
            converted_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            converted_state_dict["encoder.conv_in.bias"] = value
        # Handle decoder conv1
        elif key == "decoder.conv1.weight":
            converted_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            converted_state_dict["decoder.conv_in.bias"] = value
        # Handle encoder downsamples
        elif key.startswith("encoder.downsamples."):
            # Convert to down_blocks
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Convert residual block naming but keep the original structure
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")

            converted_state_dict[new_key] = value

        # Handle decoder upsamples
        elif key.startswith("decoder.upsamples."):
            # Convert to up_blocks
            parts = key.split(".")
            block_idx = int(parts[2])

            # Group residual blocks
            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    new_block_idx = 0
                    resnet_idx = block_idx
                elif block_idx in [4, 5, 6]:
                    new_block_idx = 1
                    resnet_idx = block_idx - 4
                elif block_idx in [8, 9, 10]:
                    new_block_idx = 2
                    resnet_idx = block_idx - 8
                elif block_idx in [12, 13, 14]:
                    new_block_idx = 3
                    resnet_idx = block_idx - 12
                else:
                    # Keep as is for other blocks
                    converted_state_dict[key] = value
                    continue

                # Convert residual block naming
                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight"
                else:
                    new_key = key

                converted_state_dict[new_key] = value

            # Handle shortcut connections
            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")

                converted_state_dict[new_key] = value

            # Handle upsamplers
            elif ".resample." in key or ".time_conv." in key:
                if block_idx == 3:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.0.upsamplers.0")
                elif block_idx == 7:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.1.upsamplers.0")
                elif block_idx == 11:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.2.upsamplers.0")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")

                converted_state_dict[new_key] = value
            else:
                new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                converted_state_dict[new_key] = value
        else:
            # Keep other keys unchanged
            converted_state_dict[key] = value

    return converted_state_dict


def extract_2d_keys(state_dict):

    # 筛选二维张量的键名
    two_d_keys = [
        key
        for key, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and "bias" not in key and "shared" not in key
    ]

    return two_d_keys


def safe_param_extract(model):
    state_dict = model.state_dict()
    safe_dict = {}

    for k, v in state_dict.items():
        # 转换为纯张量 + 断开计算图
        safe_dict[k] = v.detach().clone().cpu()

    return safe_dict


def torch2paddle(torch_state_dict):
    # torch_state_dict = load_file(torch_path)
    fc_names = extract_2d_keys(torch_state_dict)
    # fc_names = ["k","q","v","wi_0","wi_1","wo"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().to(torch.float32).numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:  # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    return paddle_state_dict


class PaddleWanVaeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (folder_paths.get_filename_list("vae"),)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "🚢 paddlemix/ppdiffusers/input"

    def load_vae(self, vae_name):
        vae_config = AutoencoderKLWan.load_config("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae")
        vae_path = folder_paths.get_full_path("vae", vae_name)
        torch_comfyui_dict = load_file(vae_path)
        torch_dict = convert_wan_vae_to_diffusers(torch_comfyui_dict)
        paddle_state_dict = torch2paddle(torch_dict)
        vae = AutoencoderKLWan(**vae_config)
        vae.set_dict(paddle_state_dict)
        # faster_set_state_dict(vae, paddle_state_dict)
        return (vae,)


class PaddleTextEncodersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_encoders_name": (folder_paths.get_filename_list("text_encoders"),)}}

    RETURN_TYPES = ("TEXTENCODER",)
    RETURN_NAMES = ("text_encoders",)
    FUNCTION = "load_text_encoders"
    CATEGORY = "🚢 paddlemix/ppdiffusers/input"

    def load_text_encoders(self, text_encoders_name):
        text_encoders_path = folder_paths.get_full_path("text_encoders", text_encoders_name)
        torch_comfyui_dict = load_file(text_encoders_path)
        paddle_state_dict = torch2paddle(torch_comfyui_dict)
        config = UMT5WANConfig()
        text_encoders = UMT5EncoderModel(config=config)
        faster_set_state_dict(text_encoders, paddle_state_dict)
        return (text_encoders,)


class PaddleWanT2VDiffusionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"diffusion_models_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_models"
    CATEGORY = "🚢 paddlemix/ppdiffusers/input"

    def load_diffusion_models(self, diffusion_models_name):
        wan_config = WanTransformer3DModel.load_config("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer")
        diffusion_models_path = folder_paths.get_full_path("diffusion_models", diffusion_models_name)
        torch_comfyui_dict = load_file(diffusion_models_path)
        torch_dict = convert_wan_transformer_to_diffusers(torch_comfyui_dict)
        paddle_state_dict = torch2paddle(torch_dict)
        diffusion_models = WanTransformer3DModel(**wan_config)
        diffusion_models.set_dict(paddle_state_dict)
        # faster_set_state_dict(diffusion_models, paddle_state_dict)
        return (diffusion_models,)


class PaddleWanText2VideoPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_model": ("MODEL",),
                "text_encoders": ("TEXTENCODER",),
                "vae": ("VAE",),
                "prompt": ("PROMPT",),
                "negative_prompt": ("PROMPT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999999999999999999999}),
                "flow_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "🚢 paddlemix/ppdiffusers/pipelines"

    def sample(
        self,
        wan_model,
        text_encoders,
        vae,
        prompt,
        negative_prompt,
        seed,
        flow_shift,
    ):
        paddle.seed(seed)
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        tokenizer = T5Tokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="tokenizer")
        pipe = WanPipeline(
            vae=vae, text_encoder=text_encoders, tokenizer=tokenizer, transformer=wan_model, scheduler=scheduler
        )

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
        ).frames[0]

        return (output,)


class PddleSaveWAN:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "lossless": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                "method": (list(s.methods.keys()),),
                # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "🚢 paddlemix/ppdiffusers/output"

    def save_images(
        self, images, fps, filename_prefix, lossless, quality, method, num_frames=0, prompt=None, extra_pnginfo=None
    ):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results: list[FileLocator] = []
        pil_images = []
        for image in images:
            i = 255.0 * image
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010F
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(
                os.path.join(full_output_folder, file),
                save_all=True,
                duration=int(1000.0 / fps),
                append_images=pil_images[i + 1 : i + num_frames],
                exif=metadata,
                lossless=lossless,
                quality=quality,
                method=method,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        animated = num_frames != 1
        return {"ui": {"images": results, "animated": (animated,)}}


NODE_CLASS_MAPPINGS = {
    "PaddleWanVaeLoader": PaddleWanVaeLoader,
    "PaddleTextEncodersLoader": PaddleTextEncodersLoader,
    "PaddleWanT2VDiffusionLoader": PaddleWanT2VDiffusionLoader,
    "PaddleWanText2VideoPipe": PaddleWanText2VideoPipe,
    "PddleSaveWAN": PddleSaveWAN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleWanVaeLoader": "Paddle Wan Vae Loader",
    "PaddleTextEncodersLoader": "Paddle TextEncoders Loader",
    "PaddleWanT2VDiffusionLoader": "Paddle Wan T2V Diffusion Loader",
    "PaddleWanText2VideoPipe": "Paddle Wan Text2Video Pipe",
    "PddleSaveWAN": "Paddle Save WAN",
}
