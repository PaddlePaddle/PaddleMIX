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


pipe = MochiPipeline.from_pretrained("/home/lcjgrp/lizhijun/llm/mochi-1-preview-pd", 
                                     variant="bf16", 
                                     paddle_dtype=paddle.bfloat16,
                                     low_cpu_mem_usage=True, 
                                     map_location="cpu"
                                     )

# 启用 VAE tiling
# pipe.enable_vae_tiling()

print("====== 模型加载后参数类型检查 ======")
# 检查主要组件的参数类型
for component_name in ['transformer', 'text_encoder', 'vae', 'scheduler']:
    if hasattr(pipe, component_name):
        component = getattr(pipe, component_name)
        print(f"\n{component_name} 组件类型: {type(component)}")
        
        if hasattr(component, 'named_parameters'):
            # 只打印部分参数，防止输出过多
            param_count = 0
            for name, param in component.named_parameters():
                if param_count < 5:  # 限制每个组件只打印5个参数
                    print(f"  - {name} 类型: {param.dtype}, 形状: {param.shape}")
                param_count += 1
            print(f"  总共 {param_count} 个参数")



prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=30).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
