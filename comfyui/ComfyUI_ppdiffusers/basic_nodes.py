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

import base64
import json
import os
from io import BytesIO

import folder_paths
import requests
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class PaddleSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.serving_web_host = os.environ.get("AISTUDIO_MS_SERVING_WEB_HOST")
        self.serving_app_token = os.environ.get("AISTUDIO_MS_AIGC_APP_JWT")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",), 
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "censor": ("BOOLEAN", {"default": True})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "🚢 paddlemix/ppdiffusers/output"

    def censor_image(self, image):
        if self.serving_web_host is None or self.serving_app_token is None:
            return True
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = (
            f"http://{self.serving_web_host}/serving/web/aigc/censor/image?serving_app_token={self.serving_app_token}"
        )
        data = {"image": img_str}
        response = requests.post(url, json=data).json()
        print(response)
        return response["result"]["pass"]

    def save_images(self, images, censor=True, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        if isinstance(images[0], Image.Image):
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].size[1], images[0].size[0]
            )
        else:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        results = list()
        for (batch_number, image) in enumerate(images):
            if not isinstance(image, Image.Image):
                img = Image.fromarray(image)
            else:
                img = image
            if censor:
                pass_censor = self.censor_image(img)
            else:
                pass_censor = True
            # breakpoint()
            if pass_censor:
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            else:
                results.append({"filename": "forbidden.png", "subfolder": "", "type": "output"})
            counter += 1

        return_dict = {"ui": {"images": results}}
        return return_dict


class PromptInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "encode"
    CATEGORY = "🚢 paddlemix/ppdiffusers/input"

    def encode(self, prompt):
        # TODO: add check for prompt
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "PaddleSaveImage": PaddleSaveImage,
    "PromptInput": PromptInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptInput": "Paddle Prompt Input",
    "PaddleSaveImage": "Paddle Save Image",
}
