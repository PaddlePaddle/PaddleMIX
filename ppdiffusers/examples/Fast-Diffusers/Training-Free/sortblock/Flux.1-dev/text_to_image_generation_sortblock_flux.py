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
from forwards import SortTaylor_forward

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42
prompt = "An image of a squirrel in Picasso style"

pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
# pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TaylorSeer settings
pipeline.transformer.__class__.num_steps = num_inference_steps

pipeline.transformer.__class__.forward = SortTaylor_forward

pipeline.transformer.current_block_residual = [None] * len(pipeline.transformer.transformer_blocks)
pipeline.transformer.current_block_encoder_residual = [None] * len(pipeline.transformer.transformer_blocks)
pipeline.transformer.current_single_block_residual = [None] * len(pipeline.transformer.single_transformer_blocks)
pipeline.transformer.previous_block_residual = [None] * len(pipeline.transformer.transformer_blocks)
pipeline.transformer.previous_single_block_residual = [None] * len(pipeline.transformer.single_transformer_blocks)
pipeline.transformer.previous_encoder_block_residual = [None] * len(pipeline.transformer.single_transformer_blocks)
pipeline.transformer.result_list = []
pipeline.transformer.result_single_list = []
pipeline.transformer.start = 900
pipeline.transformer.end = 50
pipeline.transformer.percentage = 1
pipeline.transformer.step_Num = 1
pipeline.transformer.step_Num2 = 5
pipeline.transformer.beta = 0.1
pipeline.transformer.count = 0


start_time = time.time()
prompt = "A cat holding a sign that says hello world"

image = pipeline(
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
####
start_time = time.time()
prompt = "An image of a squirrel in Picasso style"
image = pipeline(
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

image.save("text_to_image_generation-flux-dev-result.png")
