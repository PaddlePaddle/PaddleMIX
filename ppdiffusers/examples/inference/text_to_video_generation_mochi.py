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
from ppdiffusers import MochiPipeline
from ppdiffusers.utils import export_to_video


pipe = MochiPipeline.from_pretrained("/data/home/lizhijun/llm/flux-hf/models/mochi-1-preview-pd", variant="bf16", torch_dtype=paddle.bfloat16,
                                     low_cpu_mem_usage=True, map_location="cpu")

# Enable memory savings
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_tiling()

# 移动到 GPU
pipe = pipe.to("cuda")

# 启用 VAE tiling
pipe.enable_vae_tiling()

# 清理 GPU 缓存
# torch.cuda.empty_cache()


prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
