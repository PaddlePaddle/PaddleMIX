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
""" Logits Processor Helper class for Emu3. """
from typing import List, Optional, Union

import paddle
from paddlenlp.generation import LogitsProcessor, StoppingCriteria


class Emu3PrefixConstrainedLogitsHelper:
    def __init__(
        self,
        height,
        width,
        img_token,
        eoi_token,
        eos_token,
        eol_token,
        eof_token,
        pad_token,
        visual_tokens,
    ):
        self.height = height
        self.width = width
        self.img_token = img_token
        self.eoi_token = eoi_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.pad_token = pad_token
        self.visual_tokens = visual_tokens

        self.offset_cache = {}

    def __call__(self, batch_id, input_ids):
        if batch_id not in self.offset_cache:
            position = paddle.nonzero(input_ids == self.img_token, as_tuple=True)[0][0]
            self.offset_cache[batch_id] = position

        height = self.height[batch_id] if self.height.shape[0] > 1 else self.height[0]
        width = self.width[batch_id] if self.width.shape[0] > 1 else self.width[0]

        offset = input_ids.shape[0] - self.offset_cache[batch_id]
        # height = height.to(offset.device)
        # width = width.to(offset.device)

        if offset % (width + 1) == 0:
            return (self.eol_token,)
        elif offset == (width + 1) * height + 1:
            return (self.eof_token,)
        elif offset == (width + 1) * height + 2:
            return (self.eoi_token,)
        elif offset == (width + 1) * height + 3:
            return (self.eos_token,)
        elif offset > (width + 1) * height + 3:
            return (self.pad_token,)
        else:
            return self.visual_tokens


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_id: Union[int, List[int], paddle.Tensor]):
        if not isinstance(eos_token_id, paddle.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = paddle.to_tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> paddle.Tensor:
        self.eos_token_id = self.eos_token_id.to(input_ids.place)
        # is_done = isin_mps_friendly(input_ids[:, -1], self.eos_token_id)
        is_done = paddle.isin(input_ids[:, -1], self.eos_token_id)
        return is_done


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`PreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.


    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    ```
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[paddle.Tensor] = None,
        unconditional_attention_mask: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = paddle.ones_like(
                    self.unconditional_context["input_ids"], dtype=paddle.int64
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = paddle.concat(
                [
                    self.unconditional_context["attention_mask"],
                    paddle.ones_like(input_ids[:, -1:], dtype=paddle.int64),
                ],
                axis=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = paddle.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], axis=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
            return_dict=True,
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        scores = paddle.nn.functional.log_softmax(scores, axis=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        unconditional_logits = paddle.nn.functional.log_softmax(logits[:, -1], axis=-1)
        scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return scores_processed
