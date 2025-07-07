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

import paddle
from forwards import (
    taylorseer_flux_double_block_forward,
    taylorseer_flux_forward,
    taylorseer_flux_single_block_forward,
)

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42
prompt = "An image of a squirrel in Picasso style"
#
pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
# pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TaylorSeer settings
pipeline.transformer.__class__.num_steps = num_inference_steps

pipeline.transformer.__class__.forward = taylorseer_flux_forward

for double_transformer_block in pipeline.transformer.transformer_blocks:
    double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward

for single_transformer_block in pipeline.transformer.single_transformer_blocks:
    single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward


parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
# start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)
for i in range(2):
    start.record()
    img = pipeline(
        prompt, num_inference_steps=num_inference_steps, generator=paddle.Generator("cpu").manual_seed(seed)
    ).images[0]

    end.record()
    paddle.device.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3
    peak_memory = paddle.device.cuda.max_memory_allocated()

    img.save("{}.png".format("taylorseer_" + prompt))

    print(
        f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
    )
