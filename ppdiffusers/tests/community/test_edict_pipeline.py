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

from io import BytesIO

import paddle
import PIL
import requests
from IPython.display import display

from ppdiffusers import DDIMScheduler, DiffusionPipeline
from ppdiffusers.transformers import CLIPTextModel
from ppdiffusers.utils.testing_utils import get_examples_pipeline


def center_crop_and_resize(im):

    width, height = im.size
    d = min(width, height)
    left = (width - d) / 2
    upper = (height - d) / 2
    right = (width + d) / 2
    lower = (height + d) / 2

    return im.crop((left, upper, right, lower)).resize((512, 512))


paddle_dtype = paddle.float16

# scheduler and text_encoder param values as in the paper
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False,
    clip_sample=False,
)

text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path="openai/clip-vit-large-patch14",
    paddle_dtype=paddle_dtype,
)

# initialize pipeline
pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
    custom_pipeline=get_examples_pipeline("edict_pipeline"),
    # Client Error: Not Found for url: https://bj.bcebos.com/paddlenlp/models/community/CompVis/stable-diffusion-v1-4/fp16/model_index.json
    # revision="fp16",
    scheduler=scheduler,
    text_encoder=text_encoder,
    leapfrog_steps=True,
    paddle_dtype=paddle_dtype,
)

# download image
image_url = "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/imagenet_dog_1.jpeg"
response = requests.get(image_url)
image = PIL.Image.open(BytesIO(response.content))

# preprocess it
cropped_image = center_crop_and_resize(image)

# define the prompts
base_prompt = "A dog"
target_prompt = "A golden retriever"

# run the pipeline
result_image = pipeline(
    base_prompt=base_prompt,
    target_prompt=target_prompt,
    image=cropped_image,
)

for i, img in enumerate(result_image):
    img.save(f"edict_pipeline_{i}.png")
display(result_image)
