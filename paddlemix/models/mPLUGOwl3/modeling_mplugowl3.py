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

from threading import Thread

import paddle
import paddle.nn as nn
from paddlenlp.generation import TextIteratorStreamer
from paddlenlp.transformers import Qwen2PretrainedModel

from ...processors.mplugowl3_processing import (
    mPLUGOwl3ImageProcessor,
    mPLUGOwl3Processor,
)
from .configuration_mplugowl3 import mPLUGOwl3Config
from .modeling_hyper_qwen2 import HyperQwen2ForCausalLM
from .modeling_navit_siglip import SigLipVisionTransformer


class mPLUGOwl3PreTrainedModel(Qwen2PretrainedModel):
    config_class = mPLUGOwl3Config
    _no_split_modules = ["HyperQwen2DecoderLayer", "SiglipVisionTransformer"]


class mPLUGOwl3Model(mPLUGOwl3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = HyperQwen2ForCausalLM(config)
        self.vision_model = self.init_vision_module()
        self.vision_dim = self.vision_model.embed_dim
        self.embed_dim = self.config.hidden_size
        self.vision2text_model = nn.Sequential(
            nn.Linear(self.vision_dim, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.processor = None
        self.terminators = ["<|im_end|>", "<|endoftext|>"]
        self.vision_batch_size = config.vision_batch_size

    def init_vision_module(self):
        self.config.vision_config._attn_implementation = "flash_attention_2"
        model = SigLipVisionTransformer(self.config.vision_config)
        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def _small_batched_forward(self, pixel_values):
        vision_batch_size = self.vision_batch_size
        image_forward_out = []
        B = len(pixel_values)
        for i in range(0, B, vision_batch_size):
            start_idx = i
            end_idx = min(B, i + vision_batch_size)
            tmp_hs = self.vision_model(pixel_values[start_idx:end_idx], output_hidden_states=True).hidden_states[-2]
            image_forward_out.append(tmp_hs)

        vision_embedding = paddle.concat(image_forward_out, axis=0)
        assert vision_embedding.shape[0] == B
        return vision_embedding

    def forward_image(self, pixel_values):
        if pixel_values is None:
            return None
        dtype = self.language_model.model.embed_tokens.weight.dtype
        image_embeds = self._small_batched_forward(pixel_values.to(dtype))

        if self.vision2text_model is not None:
            image_embeds = self.vision2text_model(image_embeds)
        else:
            pass

        return image_embeds

    def forward(self, pixel_values=None, **kwargs):
        image_embeds = self.forward_image(pixel_values)

        return self.language_model(image_embeds=image_embeds, **kwargs)

    def _decode(self, input_ids, image_embeds, media_offset, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]

        # Note: must add position_ids, paddlenlp bug
        batch_size, seq_length = input_ids.shape
        position_ids = paddle.arange(seq_length).expand((batch_size, seq_length))

        output = self.language_model.generate(
            input_ids=input_ids,
            image_embeds=image_embeds,
            media_offset=media_offset,
            pad_token_id=0,
            eos_token_id=terminators,
            position_ids=position_ids,  # Note: must add position_ids
            attention_mask=attention_mask,
            **kwargs,
        )[0]
        # output = output[:,input_ids.shape[1]:] # paddle no need this
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, input_ids, image_embeds, media_offset, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "media_offset": media_offset,
            "pad_token_id": 0,
            "eos_token_id": terminators,
            "streamer": streamer,
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.language_model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def init_processor(self, tokenizer):
        ip = mPLUGOwl3ImageProcessor(image_size=378)
        self.processor = mPLUGOwl3Processor(image_processor=ip, tokenizer=tokenizer)
        processor = self.processor
        return processor

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        media_offset=None,
        attention_mask=None,
        tokenizer=None,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None

        with paddle.no_grad():
            image_embeds = self.forward_image(pixel_values)

            if stream:
                result = self._decode_stream(
                    input_ids=input_ids,
                    image_embeds=image_embeds,
                    media_offset=media_offset,
                    tokenizer=tokenizer,
                    **kwargs,
                )
            else:
                result = self._decode(
                    input_ids=input_ids,
                    image_embeds=image_embeds,
                    media_offset=media_offset,
                    tokenizer=tokenizer,
                    attention_mask=attention_mask,
                    decode_text=decode_text,
                    **kwargs,
                )

        return result

    def chat(
        self,
        images,
        videos,
        messages,
        tokenizer,
        processor=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt="",
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        cut_flag = kwargs.get("kwargs", True)
        if processor is None:
            if self.processor is None:
                processor = self.init_processor(tokenizer)
            else:
                processor = self.processor
        inputs = processor(messages, images=images, videos=videos, cut_enable=cut_flag)
        inputs.update(
            {
                "tokenizer": tokenizer,
                "max_new_tokens": max_new_tokens,
                # 'stream':True,
            }
        )
        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                # "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                # "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        with paddle.no_grad():
            res = self.generate(**inputs, stream=stream, decode_text=True, **generation_config)

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            return stream_gen()

        else:
            answer = res[0]
            return answer
