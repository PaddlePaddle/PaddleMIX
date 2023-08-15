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

from ppdiffusers import LVDMUncondPipeline

# 加载模型和scheduler
pipe = LVDMUncondPipeline.from_pretrained("westfish/lvdm_short_sky_epoch2239_step150079")

# 执行pipeline进行推理
seed = 1000
generator = paddle.Generator().manual_seed(seed)
samples = pipe(
    batch_size=1,
    num_frames=4,
    num_inference_steps=50,
    generator=generator,
    eta=1,
    save_dir=".",
    save_name="ddim_lvdm_short_sky_epoch2239_step150079",
    scale_factor=0.33422927,
    shift_factor=1.4606637,
)
