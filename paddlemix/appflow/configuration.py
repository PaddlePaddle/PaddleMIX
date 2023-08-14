# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .image2image_text_guided_generation import (StableDiffusionImg2ImgTask,
                                                 StableDiffusionUpscaleTask)
from .image2text_generation import Blip2CaptionTask
from .openset_det_sam import OpenSetDetTask, OpenSetSegTask
from .text2image_generation import (StableDiffusionTask,
                                    VersatileDiffusionDualGuidedTask)
from .text2image_inpaiting import StableDiffusionInpaintTask
from .text2text_generation import ChatGlmTask
from .text2video_generation import TextToVideoSDTask

APPLICATIONS = {
    "openset_det_sam": {
        "models": {
            "GroundingDino/groundingdino-swint-ogc": {
                "task_class": OpenSetDetTask,
                "task_flag": "openset_det_sam-groundingdino",
            },
            "Sam/SamVitH-1024": {
                "task_class": OpenSetSegTask,
                "task_flag": "openset_det_sam-SamVitH-1024",
            },
        },
        "default": {
            "model": "GroundingDino/groundingdino-swint-ogc",
        },
    },
    "inpainting": {
        "models": {
            "THUDM/chatglm-6b": {
                "task_class": ChatGlmTask,
                "task_flag": "inpainting-chatglm-6b",
            },
            "GroundingDino/groundingdino-swint-ogc": {
                "task_class": OpenSetDetTask,
                "task_flag": "inpainting-groundingdino",
            },
            "Sam/SamVitH-1024": {
                "task_class": OpenSetSegTask,
                "task_flag": "inpainting-SamVitH-1024",
            },
            "stabilityai/stable-diffusion-2-inpainting": {
                "task_class": StableDiffusionInpaintTask,
                "task_flag": "inpainting-stable-diffusion-2",
            },
        },
        "default": {
            "model": "GroundingDino/groundingdino-swint-ogc",
        },
    },
    "auto_label": {
        "models": {
            "paddlemix/blip2-caption-opt2.7b": {
                "task_class": Blip2CaptionTask,
                "task_flag": "autolabel_blip2-caption-opt2.7b",
            },
            "GroundingDino/groundingdino-swint-ogc": {
                "task_class": OpenSetDetTask,
                "task_flag": "openset_det_sam-groundingdino",
            },
            "Sam/SamVitH-1024": {
                "task_class": OpenSetSegTask,
                "task_flag": "openset_det_sam-SamVitH-1024",
            },
        },
    },
    "text2image_generation": {
        "models": {
            "stabilityai/stable-diffusion-2": {
                "task_class": StableDiffusionTask,
                "task_flag": "text2image_generation-stable-diffusion-2",
            }
        },
        "default": {
            "model": "stabilityai/stable-diffusion-2",
        },
    },
    "image2image_text_guided_generation": {
        "models": {
            "Linaqruf/anything-v3.0": {
                "task_class": StableDiffusionImg2ImgTask,
                "task_flag":
                "image2image_text_guided_generation-Linaqruf/anything-v3.0",
            }
        },
        "default": {
            "model": "Linaqruf/anything-v3.0",
        },
    },
    "image2image_text_guided_upscaling": {
        "models": {
            "stabilityai/stable-diffusion-x4-upscaler": {
                "task_class": StableDiffusionUpscaleTask,
                "task_flag":
                "image2image_text_guided_upscaling-stabilityai/stable-diffusion-x4-upscaler",
            }
        },
        "default": {
            "model": "stabilityai/stable-diffusion-x4-upscaler",
        },
    },
    "dual_text_and_image_guided_generation": {
        "models": {
            "shi-labs/versatile-diffusion": {
                "task_class": VersatileDiffusionDualGuidedTask,
                "task_flag":
                "dual_text_and_image_guided_generation-shi-labs/versatile-diffusion",
            }
        },
        "default": {
            "model": "shi-labs/versatile-diffusion",
        },
    },
    "text_to_video_generation": {
        "models": {
            "damo-vilab/text-to-video-ms-1.7b": {
                "task_class": TextToVideoSDTask,
                "task_flag":
                "text_to_video_generation-damo-vilab/text-to-video-ms-1.7b",
            }
        },
        "default": {
            "model": "damo-vilab/text-to-video-ms-1.7b",
        },
    },
}
