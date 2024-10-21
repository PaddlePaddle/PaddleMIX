# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ldm.model_b import ModelB
from ldm.model_c import ModelC

model = ModelC()
model.eval()
model_b = ModelB()
model_b.eval()

caption = "A beauty girl in winter"
batch_size = 2
seed = 1000
image, latent_c = model.log_image(caption, seed=seed, batch_size=batch_size)
image_b, latent_b = model_b.log_image(latent_c, caption, seed=seed, batch_size=batch_size)
from PIL import Image

for i, image_i in enumerate(image_b):
    Image.fromarray(image_i).save("./result_{}.png".format(i))
