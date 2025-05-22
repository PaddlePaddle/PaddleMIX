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
import os

import numpy as np
import paddle
import PIL.Image
from paddlenlp.transformers import LlamaTokenizerFast
from tqdm import tqdm

from paddlemix.models.janus import JanusMultiModalityCausalLM
from paddlemix.processors import JanusImageProcessor, JanusVLChatProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-1.3B")
parser.add_argument(
    "--prompt",
    type=str,
    default="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
)
parser.add_argument("--dtype", type=str, default="float16")
args = parser.parse_args()

vl_gpt = JanusMultiModalityCausalLM.from_pretrained(args.model_path, dtype=args.dtype)
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)
image_processer = JanusImageProcessor.from_pretrained(args.model_path)
vl_chat_processor: JanusVLChatProcessor = JanusVLChatProcessor(image_processer, tokenizer)

conversation = [
    {
        "role": "User",
        "content": args.prompt,
    },
    {"role": "Assistant", "content": ""},
]
sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
)
prompt = sft_format + vl_chat_processor.image_start_tag


@paddle.no_grad()
def generate(
    mmgpt,
    vl_chat_processor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 2,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = paddle.to_tensor(data=input_ids.input_ids, dtype="int64")
    tokens = paddle.zeros(shape=(parallel_size * 2, len(input_ids)), dtype="int32")
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)  # [4, 50, 2048]
    generated_tokens = paddle.zeros(shape=(parallel_size, image_token_num_per_image), dtype="int32")
    batch_size, seq_length = inputs_embeds.shape[:2]
    for i in tqdm(range(image_token_num_per_image)):
        batch_size, seq_length = inputs_embeds.shape[:2]

        past_key_values_length = outputs.past_key_values[0][0].shape[1] if i != 0 else 0
        position_ids = paddle.arange(past_key_values_length, seq_length + past_key_values_length).expand(
            (batch_size, seq_length)
        )

        outputs = mmgpt.language_model.llama(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,  # [4, 1, 2048]
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = paddle.nn.functional.softmax(x=logits / temperature, axis=-1)
        next_token = paddle.multinomial(x=probs, num_samples=1)

        generated_tokens[:, i] = next_token.squeeze(axis=-1)
        next_token = paddle.concat(x=[next_token.unsqueeze(axis=1), next_token.unsqueeze(axis=1)], axis=1).reshape(
            [-1]
        )
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(axis=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype="int32"), shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to("float32").cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    os.makedirs("janus_generated_samples", exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join("janus_generated_samples", "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


generate(vl_gpt, vl_chat_processor, prompt)
