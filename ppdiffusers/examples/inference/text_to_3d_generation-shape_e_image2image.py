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

import paddle

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import export_to_gif, load_image

repo = "openai/shap-e-img2img"
pipe = DiffusionPipeline.from_pretrained(repo, paddle_dtype=paddle.float16)
guidance_scale = 3.0
image_url = "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-upgrade0193/shap-e_corgi.png"
image = load_image(image_url).convert("RGB")
images = pipe(
    image,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images
gif_path = export_to_gif(images[0], "text_to_3d_generation-shape_e_image2image-result-corgi_3d.gif")
