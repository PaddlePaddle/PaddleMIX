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

import time

import paddle
from forwards.teablockcache_taylor_flux_forward import TeaBlockCacheTaylorForward

from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

# Generation parameters
num_inference_steps = 50
seed = 42

# Load pipeline
pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
# pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TeaBlockCache + Taylor settings
FluxTransformer2DModel.forward = TeaBlockCacheTaylorForward

# Configure TeaBlockCache parameters
pipeline.transformer.cnt = 0
pipeline.transformer.num_steps = num_inference_steps
pipeline.transformer.step_start = 50
pipeline.transformer.step_end = 950
pipeline.transformer.block_cache_start = 1
pipeline.transformer.single_block_cache_start = 1
pipeline.transformer.block_rel_l1_thresh = 2
pipeline.transformer.single_block_rel_l1_thresh = 2

# Initialize state dictionaries
pipeline.transformer.block_heuristic_states = {}
pipeline.transformer.single_block_heuristic_states = {}

# Initialize Taylor cache system
pipeline.transformer.enable_teacache = True
pipeline.transformer.rel_l1_thresh = 2
pipeline.transformer.taylor_cache_system = {
    "max_order": 1,
    "first_enhance": 1,
    "cache": {"hidden": {}},
    "activated_steps": [],
    "step_counter": 0,
}

# Generate first image
start_time = time.time()
prompt = "A cat holding a sign that says hello world"

image = pipeline(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(seed),
).images[0]
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
image.save("text_to_image_generation-teablockcache-taylor-flux-dev-result-0.png")

# Generate second image
start_time = time.time()
prompt = "An image of a squirrel in Picasso style"
image = pipeline(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(seed),
).images[0]
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
image.save("text_to_image_generation-teablockcache-taylor-flux-dev-result-1.png")

# Report cache statistics
if hasattr(pipeline.transformer, "block_heuristic_states"):
    num_cached_blocks = len(pipeline.transformer.block_heuristic_states)
    print(f"Transformer blocks cached: {num_cached_blocks}")

if hasattr(pipeline.transformer, "single_block_heuristic_states"):
    num_cached_single_blocks = len(pipeline.transformer.single_block_heuristic_states)
    print(f"Single blocks cached: {num_cached_single_blocks}")

# Report Taylor cache statistics
if hasattr(pipeline.transformer, "taylor_cache_system"):
    taylor_steps = len(pipeline.transformer.taylor_cache_system["activated_steps"])
    taylor_cache_size = len(pipeline.transformer.taylor_cache_system["cache"]["hidden"])
    print(f"Taylor cache activated steps: {taylor_steps}")
    print(f"Taylor cache coefficients stored: {taylor_cache_size}")
