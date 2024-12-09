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

# from paddlenlp.transformers import AutoTokenizer
from PIL import Image

from paddlemix.models.minicpm_v.modeling_minicpmv import MiniCPMV
from paddlemix.models.minicpm_v.tokenization_minicpmv_fast import MiniCPMVTokenizerFast

from decord import VideoReader, cpu    # pip install decord
MODEL_NAME = "openbmb/MiniCPM-V-2_6"
model = MiniCPMV.from_pretrained(MODEL_NAME, dtype="bfloat16")
model = model.eval()
tokenizer = MiniCPMVTokenizerFast.from_pretrained(MODEL_NAME)

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path="paddlemix/demo_images/red-panda.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]}, 
]

# Set decode params for video
params = {}
params["use_image_id"] = False
params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    max_new_tokens=2048,  # 2048
    **params
)
print(res)