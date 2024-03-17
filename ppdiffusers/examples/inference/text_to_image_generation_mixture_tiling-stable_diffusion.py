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
import os

from ppdiffusers import DiffusionPipeline, LMSDiscreteScheduler

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    custom_pipeline=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "community/mixture_tiling.py")),
)

# Mixture of Diffusers generation
image = pipeline(
    prompt=[
        [
            "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
            "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
            "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        ]
    ],
    tile_height=640,
    tile_width=640,
    tile_row_overlap=0,
    tile_col_overlap=256,
    guidance_scale=8,
    seed=7178915308,
    num_inference_steps=50,
)["images"][0]
image.save("text_to_image_generation_mixture_tiling-stable_diffusion-mixture_tiling" + ".png")
