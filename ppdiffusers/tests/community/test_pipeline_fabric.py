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
import requests
from PIL import Image

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils.testing_utils import get_examples_pipeline

# load the pipeline
model_id_or_path = "runwayml/stable-diffusion-v1-5"
# can also be used with dreamlike-art/dreamlike-photoreal-2.0
pipe = DiffusionPipeline.from_pretrained(
    model_id_or_path, paddle_dtype=paddle.float16, custom_pipeline=get_examples_pipeline("pipeline_fabric")
)

# let's specify a prompt
prompt = "An astronaut riding an elephant"
negative_prompt = "lowres, cropped"

# call the pipeline
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    generator=paddle.Generator().manual_seed(17),
).images[0]

image.save("horse_to_elephant.jpg")

# let's try another example with feedback
url = "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/Ablackcoloredcar.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "photo, A blue colored car, fish eye"
liked = [init_image]
# same goes with disliked

# call the pipeline
paddle.seed(seed=0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    liked=liked,
    num_inference_steps=20,
    generator=paddle.Generator().manual_seed(0),
).images[0]

image.save("black_to_blue.png")
