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

from ppdiffusers import DiTPipeline, DPMSolverMultistepScheduler

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=paddle.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# pick words from Imagenet class labels
pipe.labels  # to print all available words

# pick words that exist in ImageNet
words = ["white shark", "umbrella"]

class_ids = pipe.get_label_ids(words)

generator = paddle.Generator().manual_seed(33)
output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

image = output.images[0]  # label 'white shark'

image.save("text_to_image_generation_dit.png")
