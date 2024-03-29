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

from ppdiffusers import ConsistencyModelPipeline

# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, paddle_dtype=paddle.float16)

# Onestep Sampling
image = pipe(num_inference_steps=1).images[0]
image.save("cd_imagenet64_l2_onestep_sample.png")

# Onestep sampling, class-conditional image generation
# ImageNet-64 class label 145 corresponds to king penguins
image = pipe(num_inference_steps=1, class_labels=145).images[0]
image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

# Multistep sampling, class-conditional image generation
# Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo:
# https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
image.save("cd_imagenet64_l2_multistep_sample_penguin.png")
