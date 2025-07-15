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
from forwards import FirstBlock_taylor_predict_Forward

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42

prompt = "An image of a squirrel in Picasso style"
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

pipe.transformer.__class__.forward = FirstBlock_taylor_predict_Forward

pipe.transformer.enable_teacache = True
pipe.transformer.cnt = 0
pipe.transformer.num_steps = 50


pipe.transformer.residual_diff_threshold = 0.05  # 0.05  7.6s
pipe.transformer.downsample_factor = 1
pipe.transformer.accumulated_rel_l1_distance = 0
pipe.transformer.prev_first_hidden_states_residual = None
pipe.transformer.previous_residual = None


# pipe.to("cuda")

parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
# start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)

for i in range(2):
    start.record()
    img = pipe(prompt, num_inference_steps=num_inference_steps, generator=paddle.Generator().manual_seed(seed)).images[
        0
    ]

    end.record()
    paddle.device.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3
    peak_memory = paddle.device.cuda.max_memory_allocated()

    img.save("{}.png".format("1_" + "An image of a squirrel in Picasso style"))
    # img.save(f"{pkl_list[i]}.png")

    print(
        f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
    )
