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

from ppdiffusers import StableDiffusion3Pipeline
from ppdiffusers.peft import PeftModel

model_id = "stabilityai/stable-diffusion-3.5-medium"
lora_ckpt_path = "jieliu/SD3.5M-FlowGRPO-GenEval"
device = "cuda"


pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=paddle.float16)
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_ckpt_path)
pipe.transformer = pipe.transformer.merge_and_unload()
pipe = pipe.to(device)

prompt = "a photo of a black kite and a green bear"
image = pipe(prompt, height=512, width=512, num_inference_steps=40, guidance_scale=4.5, negative_prompt="").images[0]
image.save("flow_grpo.png")
