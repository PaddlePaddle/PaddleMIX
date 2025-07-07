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

from ppdiffusers import (
    FluxPipeline,
    TeaBlockCacheTaylorConfig,
    apply_teablockcache_taylor,
)

# Generation parameters
num_inference_steps = 50
seed = 42

# Load pipeline
pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

# TeaBlockCache + Taylor configuration using hook-based approach
config = TeaBlockCacheTaylorConfig(
    step_start=50,  # Start timestep for TeaBlockCache caching
    step_end=950,  # End timestep for TeaBlockCache caching
    block_cache_start=1,  # Start block index for transformer blocks caching
    single_block_cache_start=1,  # Start block index for single transformer blocks caching
    block_rel_l1_thresh=2.0,  # Relative L1 threshold for transformer blocks
    single_block_rel_l1_thresh=2.0,  # Relative L1 threshold for single transformer blocks
    taylor_max_order=1,  # Maximum order for Taylor expansion
    taylor_first_enhance=1,  # First enhance parameter for Taylor cache
    rel_l1_thresh=2.0,  # Relative L1 threshold for Taylor cache system
    num_inference_steps=num_inference_steps,  # Total number of inference steps
    current_timestep_callback=lambda: pipeline._current_timestep,  # Callback to get current timestep
)

# Apply TeaBlockCache + Taylor optimization using the hook system
apply_teablockcache_taylor(pipeline.transformer, config)

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
image.save("text_to_image_generation-teablockcache-taylor-flux-hook-result-0.png")

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
image.save("text_to_image_generation-teablockcache-taylor-flux-hook-result-1.png")

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
