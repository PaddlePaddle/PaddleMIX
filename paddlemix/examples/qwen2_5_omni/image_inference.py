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

import argparse
import json
import os

import paddle
import soundfile as sf

from paddlemix.models.qwen2_5_omni import Qwen2_5OmniModel, WhisperFeatureExtractor
from paddlemix.models.qwen2_5_omni.mm_process import process_mm_info
from paddlemix.models.qwen2_vl.mix_qwen2_tokenizer import MIXQwen2Tokenizer
from paddlemix.processors.qwen2_5_omni_processing import Qwen2_5OmniProcessor
from paddlemix.processors.qwen2_vl_processing import Qwen2VLImageProcessor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of pretrained model.")
    parser.add_argument(
        "--attn_implementation", default="sdpa", help="Attention implementation type e.g., eager, sdpa"
    )
    parser.add_argument("--compute_dtype", default="bfloat16", help="The compute dtype.")
    parser.add_argument(
        "--image_file", default="paddlemix/demo_images/examples_image1.jpg", help="Path to the input image file."
    )
    parser.add_argument("--question", default="What are in this image?", help="Input question text for the model.")
    parser.add_argument(
        "--output_audio_file", type=str, default="image_output.wav", help="Path to save the output audio file."
    )

    return parser.parse_args()


args = parse_arguments()
model_path = args.model_name_or_path
model = Qwen2_5OmniModel.from_pretrained(
    model_path, dtype=args.compute_dtype, attn_implementation=args.attn_implementation
).eval()

tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
processor_config = json.load(open(os.path.join(model_path, "preprocessor_config.json")))
whisper_proc = WhisperFeatureExtractor(**processor_config)
img_proc = Qwen2VLImageProcessor(**processor_config)
processor = Qwen2_5OmniProcessor(img_proc, whisper_proc, tokenizer)

conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": args.image_file,
            },
            {"type": "text", "text": args.question},
        ],
    },
]
# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print(text)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pd",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)

# convert dtype
if getattr(inputs, "pixel_values", None) is not None:
    inputs.pixel_values = inputs.pixel_values.astype(args.compute_dtype)

if getattr(inputs, "pixel_values_videos", None) is not None:
    inputs.pixel_values_videos = inputs.pixel_values_videos.astype(args.compute_dtype)

if getattr(inputs, "input_features", None) is not None:
    inputs.input_features = inputs.input_features.astype(args.compute_dtype)

# Inference: Generation of the output text and audio
with paddle.no_grad():
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    args.output_audio_file,
    audio.reshape([-1]).detach().cpu().numpy(),
    samplerate=24000,
)
