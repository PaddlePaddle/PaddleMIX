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
import os
import paddle

from paddlenlp.transformers import LlamaModel
from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast
from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from ppdiffusers.utils import export_to_video

os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16)
tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer = transformer,
    text_encoder = text_encoder,
    tokenizer = tokenizer,
    paddle_dtype=paddle.float16)
prompt = 'A cat walks on the grass, realistic.'
output = pipe(
    prompt=prompt,
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
    generator=paddle.Generator().manual_seed(42),
).frames[0]
export_to_video(output, "./text_to_video_generation-hunyuan_video.mp4", fps=15)