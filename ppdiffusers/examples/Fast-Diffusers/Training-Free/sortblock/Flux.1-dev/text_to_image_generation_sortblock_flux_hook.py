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

import time

import paddle

from ppdiffusers import FluxPipeline, SortBlockConfig, apply_sort_block

# Load the pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

# Configure SortBlock optimization
config = SortBlockConfig(
    num_inference_steps=50,
    timestep_start=900,
    timestep_end=100,
    percentage=1.0,
    step_num=1,
    step_num2=5,
    beta=0.3,
    current_timestep_callback=lambda: getattr(pipe, "_current_timestep", None),
)

# Apply SortBlock optimization using the integrated framework
apply_sort_block(pipe.transformer, config)

# Alternative method using enable_cache
# pipe.transformer.enable_cache(config)

# Generate images with optimization
start_time = time.time()
prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(42),
).images[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

image.save("text_to_image_generation-sortblock-flux-hook-result-0.png")

# Generate second image
start_time = time.time()
prompt = "An image of a squirrel in Picasso style"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(42),
).images[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

image.save("text_to_image_generation-sortblock-flux-hook-result-1.png")

# Disable optimization if needed
# pipe.transformer.disable_cache()
