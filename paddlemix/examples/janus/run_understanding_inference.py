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

import argparse
import base64
import io
from typing import Dict, List

import paddle
import PIL.Image
from paddlenlp.transformers import LlamaTokenizerFast

from paddlemix.models.janus import JanusMultiModalityCausalLM
from paddlemix.processors import JanusImageProcessor, JanusVLChatProcessor


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

        Support file path or base64 images.

        Args:
            conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
                [
                    {
                        "role": "User",
                        "content": "<image_placeholder>
    Extract all information from this image and convert them into markdown format.",
                        "images": ["./examples/table_datasets.png"]
                    },
                    {"role": "Assistant", "content": ""},
                ]

        Returns:
            pil_images (List[PIL.Image.Image]): the list of PIL images.

    """
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        for image_data in message["images"]:
            if image_data.startswith("data:image"):
                _, image_data = image_data.split(",", 1)
                image_bytes = base64.b64decode(image_data)
                pil_img = PIL.Image.open(io.BytesIO(image_bytes))
            else:
                pil_img = PIL.Image.open(image_data)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
    return pil_images


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-1.3B")
parser.add_argument("--image_file", type=str, required=True)
parser.add_argument("--question", type=str, default="What is shown in this image?")
parser.add_argument("--dtype", type=str, default="float16")

args = parser.parse_args()

vl_gpt = JanusMultiModalityCausalLM.from_pretrained(args.model_path, dtype=args.dtype)
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)
image_processer = JanusImageProcessor.from_pretrained(args.model_path)
vl_chat_processor: JanusVLChatProcessor = JanusVLChatProcessor(image_processer, tokenizer)

conversation = [
    {
        "role": "User",
        "content": f"<image_placeholder>\n{args.question}",
        "images": [args.image_file],
    },
    {"role": "Assistant", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
device = prepare_inputs["pixel_values"].place
prepare_inputs.to(device, dtype=args.dtype)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
bs, seq_len = prepare_inputs.attention_mask.shape
position_ids = paddle.arange(seq_len, dtype=paddle.int64).reshape([1, -1])
outputs = vl_gpt.language_model.generate(
    input_ids=prepare_inputs["input_ids"],
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,  # 512,
    do_sample=False,
    use_cache=True,
)
answer = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
