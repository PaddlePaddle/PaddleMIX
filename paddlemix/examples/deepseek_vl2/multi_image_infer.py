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
import os
import sys

from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import DeepseekTokenizerFast

from paddlemix.models.deepseek_vl2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
from paddlemix.processors.deepseek_vl2_processing import DeepseekVLV2Processor

sys.path.append(os.path.dirname(__file__))
from utils import load_pil_images

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-vl2-tiny")
parser.add_argument("--image_file_1", type=str, required=True)
parser.add_argument("--image_file_2", type=str, required=True)
parser.add_argument("--image_file_3", type=str, required=True)
parser.add_argument("--question", type=str, default="What is shown in this image?")
parser.add_argument("--dtype", type=str, default="bfloat16")

args = parser.parse_args()

model_path = args.model_path

tokenizer = DeepseekTokenizerFast.from_pretrained(model_path)
config = DeepseekVLV2Config.from_pretrained(model_path)

candidate_resolutions = config["candidate_resolutions"]
patch_size = config.vision_config["patch_size"]
downsample_ratio = config["downsample_ratio"]

vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor(
    tokenizer=tokenizer,
    candidate_resolutions=candidate_resolutions,
    patch_size=patch_size,
    downsample_ratio=downsample_ratio,
)

tokenizer = vl_chat_processor.tokenizer
vl_gpt: DeepseekVLV2ForCausalLM = DeepseekVLV2ForCausalLM.from_pretrained(model_path, dtype=args.dtype).eval()


conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\n"
        "This is image_2: <image>\n"
        f"This is image_3: <image>\n {args.question}",
        "images": [
            f"{args.image_file_1}",
            f"{args.image_file_2}",
            f"{args.image_file_3}",
        ],
    },
    {"role": "<|Assistant|>", "content": ""},
]


pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True, system_prompt=""
)
prepare_inputs.images = prepare_inputs.images.astype(args.dtype)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    trunc_input=True,
    output_attentions=True,
    use_cache=True,  # must true
    return_dict=True,
)
outputs = vl_gpt.language.generate(
    generation_config=generation_config,
    inputs_embeds=inputs_embeds.astype(args.dtype),
    attention_mask=prepare_inputs.attention_mask,
)

answer = tokenizer.decode(outputs[0][0].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
