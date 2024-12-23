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

from paddlemix.models.janus import JanusFlowMultiModalityCausalLM
from paddlemix.processors import JanusImageProcessor, JanusVLChatProcessor
from ppdiffusers.models import AutoencoderKL

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="deepseek-ai/JanusFlow-1.3B")
parser.add_argument(
    "--prompt",
    type=str,
    default="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
)
parser.add_argument("--inference_step", type=int, default=30)
parser.add_argument("--dtype", type=str, default="float16")

args = parser.parse_args()

vl_gpt = JanusFlowMultiModalityCausalLM.from_pretrained(args.model_path, dtype=args.dtype)
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)
image_processer = JanusImageProcessor.from_pretrained(args.model_path)
vl_chat_processor: JanusVLChatProcessor = JanusVLChatProcessor(image_processer, tokenizer)
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

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
    args,
    vl_gpt,
    vl_chat_processor,
    tokenizer,
    prompt,
    cfg_weight: float = 2.0,
    num_inference_steps: int = 30,
    batch_size: int = 1,
):
    input_ids = tokenizer(prompt, return_tensors="pd")["input_ids"]
    tokens = paddle.stack(x=[input_ids] * batch_size * 2)[:, 0, :]
    tokens[batch_size:, 1:] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    inputs_embeds = inputs_embeds[:, :-1, :]
    z = paddle.randn(shape=(batch_size, 4, 48, 48), dtype=args.dtype)
    dt = 1.0 / num_inference_steps
    dt = paddle.zeros_like(x=z, dtype=args.dtype) + dt
    attention_mask = paddle.ones(shape=(2 * batch_size, tuple(inputs_embeds.shape)[1] + 577))
    attention_mask[batch_size:, 1 : tuple(inputs_embeds.shape)[1]] = 0
    attention_mask = attention_mask.astype(dtype="int32")

    for step in tqdm(range(num_inference_steps)):
        z_input = paddle.concat(x=[z, z], axis=0)
        t = step / num_inference_steps * 1000.0
        t = paddle.to_tensor(data=[t] * tuple(z_input.shape)[0]).to(dt.place)
        z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
        z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
        z_emb = z_emb.reshape([tuple(z_emb.shape)[0], tuple(z_emb.shape)[1], -1]).transpose(perm=[0, 2, 1])
        z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)
        llm_emb = (
            paddle.concat(x=[inputs_embeds, t_emb.unsqueeze(axis=1), z_emb], axis=1)
            if step == 0
            else paddle.concat(x=[t_emb.unsqueeze(axis=1), z_emb], axis=1)
        )
        bs, seq_len, dim = llm_emb.shape
        past_seen_tokens = inputs_embeds.shape[1] if step != 0 else 0
        position_ids = paddle.arange(past_seen_tokens, past_seen_tokens + seq_len, dtype=paddle.int64).reshape([1, -1])
        outputs = vl_gpt.language_model.llama(
            position_ids=position_ids,
            inputs_embeds=llm_emb,
            use_cache=True,
            attention_mask=attention_mask,
            past_key_values=past_key_values if step != 0 else None,
            return_dict=True,
        )
        if step == 0:
            past_key_values = []
            for kv in outputs.past_key_values:
                # [2, 607, 16, 128]
                k, v = kv[0], kv[1]
                past_key_values.append((k[:, : inputs_embeds.shape[1], :, :], v[:, : inputs_embeds.shape[1], :, :]))
            past_key_values = tuple(past_key_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
        hidden_states = hidden_states.reshape([tuple(z_emb.shape)[0], 24, 24, 768]).transpose(perm=[0, 3, 1, 2])
        v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
        v_cond, v_uncond = paddle.chunk(x=v, chunks=2)
        v = cfg_weight * v_cond - (cfg_weight - 1.0) * v_uncond
        z = z + dt * v
    decoded_image = vae.decode(z / vae.config.scaling_factor).sample
    images = decoded_image.astype(dtype="float32").clip_(min=-1.0, max=1.0).transpose(perm=[0, 2, 3, 1]).cpu().numpy()
    images = ((images + 1) / 2.0 * 255).astype(np.uint8)

    os.makedirs("janusflow_generated_samples", exist_ok=True)
    for i in range(batch_size):
        save_path = os.path.join("janusflow_generated_samples", "img_{}.jpg".format(i))
        PIL.Image.fromarray(images[i]).save(save_path)


generate(args, vl_gpt, vl_chat_processor, tokenizer, prompt, num_inference_steps=args.inference_step)
