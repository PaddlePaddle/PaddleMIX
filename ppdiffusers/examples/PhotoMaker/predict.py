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

os.environ["USE_PEFT_BACKEND"] = "True"
# ignore warning
os.environ["GLOG_minloglevel"] = "2"

import paddle
from photomaker import PhotoMakerStableDiffusionXLPipeline

from ppdiffusers import EulerDiscreteScheduler
from ppdiffusers.utils import load_image

base_model_path = "SG161222/RealVisXL_V3.0"
photomaker_path = "TencentARC/PhotoMaker"
photomaker_ckpt = "photomaker-v1.bin"

### Load base model # noqa: E266
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,  # can change to any base model based on SDXL
    paddle_dtype=paddle.float16,
    use_safetensors=True,
    variant="fp16",
    low_cpu_mem_usage=True,
)

### Load PhotoMaker checkpoint # noqa: E266
pipe.load_photomaker_adapter(
    photomaker_path,
    weight_name=photomaker_ckpt,
    from_hf_hub=True,
    from_diffusers=True,
    trigger_word="img",  # define the trigger word
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.fuse_lora()

### define the input ID images # noqa: E266
input_folder_name = "./examples/newton_man"
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted(
    [os.path.join(input_folder_name, basename) for basename in image_basename_list if basename.endswith(".jpg")]
)

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

# Note that the trigger word `img` must follow the class word for personalization
prompt = "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality"
negative_prompt = (
    "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
)
generator = paddle.Generator().manual_seed(42)
gen_images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=50,
    start_merge_step=10,
    generator=generator,
).images[0]
gen_images.save("out_photomaker.png")
