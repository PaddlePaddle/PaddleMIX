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
from forwards import CGTaylor_flux_forward

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42

prompt = "An image of a squirrel"

prompt = "An image of a squirrel in Picasso style"
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
# pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


pipe.transformer.__class__.forward = CGTaylor_flux_forward
pipe.transformer.enable_teacache = True
pipe.transformer.cnt = 0
pipe.transformer.num_steps = num_inference_steps

pipe.transformer.pre_firstblock_hidden_states = None
pipe.transformer.previous_residual = None
pipe.transformer.pre_compute_hidden = None
pipe.transformer.predict_loss = None
pipe.transformer.predict_hidden_states = None
pipe.transformer.threshold = 0.03


parameter_peak_memory = 0


start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)


for i in range(2):
    paddle.device.cuda.reset_max_memory_allocated()

    start_time = time.time()
    img = pipe(prompt, num_inference_steps=num_inference_steps, generator=paddle.Generator().manual_seed(seed)).images[
        0
    ]
    elapsed1 = time.time() - start_time
    peak_memory = paddle.device.cuda.max_memory_allocated()

    img.save("firstblockpredict.png")
    print(
        f"epoch time: {elapsed1:.2f} sec, parameter memory: {parameter_peak_memory/(1024 * 1024 * 1024):.2f} GB, memory: {peak_memory/(1024 * 1024 * 1024):.2f} GB"
    )
