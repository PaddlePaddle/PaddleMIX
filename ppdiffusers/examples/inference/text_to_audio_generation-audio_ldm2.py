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
import scipy
from IPython.display import Audio, display

from ppdiffusers import AudioLDM2Pipeline

repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, paddle_dtype=paddle.float16)

# define the prompts
prompt = "The sound of a hammer hitting a wooden surface."
negative_prompt = "Low quality."

# set the seed for generator
generator = paddle.Generator().manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_length_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

output_path = "text_to_audio_generation-audio_ldm2-techno.wav"

# save the best audio sample (index 0) as a .wav file
scipy.io.wavfile.write(output_path, rate=16000, data=audio[0])
# 可以直接使用 IPython.display.Audio 来显示音频文件
display(Audio(output_path))
