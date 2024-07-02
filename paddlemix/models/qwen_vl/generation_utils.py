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

from typing import Iterable, List, Tuple, Union

import numpy as np
import paddle
from paddlenlp.generation import LogitsProcessor
from paddlenlp.transformers import PretrainedTokenizer

HistoryType = List[Tuple[str, str]]
TokensType = List[int]


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_context(
    tokenizer: PretrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")["input_ids"]

        def _tokenize_str(role, content):
            return (
                f"{role}\n{content}",
                tokenizer.encode(role, allowed_special=set(tokenizer.IMAGE_ST))["input_ids"]
                + nl_tokens
                + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))["input_ids"],
            )

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        raw_text = ""
        context_tokens = []
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"
            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")["input_ids"]
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)["input_ids"]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return raw_text, context_tokens


def _decode_default(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_words: List[str],
    tokenizer: PretrainedTokenizer,
    raw_text_len: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = "replace"
):
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate: ", trim_decode_tokens)
    end_reason = f"Gen length {len(tokens)}"
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f"Gen {eod_word!r}"
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nEnd Reason:", end_reason)
        print("\nGenerate: ", trim_decode_tokens)
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def _decode_chatml(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_token_ids: List[int],
    tokenizer: PretrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = "replace"
):
    end_reason = f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break
    trim_decode_tokens = tokenizer.decode(tokens[context_length:eod_token_idx], errors=errors)
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors))
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def decode_tokens(
    tokens: Union[paddle.Tensor, TokensType],
    tokenizer: PretrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = "replace",
) -> str:
    if paddle.is_tensor(x=tokens):
        tokens = tokens.cpu().numpy().tolist()
    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    elif chat_format == "raw":
        return _decode_default(
            tokens,
            stop_words=["<|endoftext|>"],
            eod_words=["<|endoftext|>"],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")


class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.

    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):
        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}.")
        if any(
            any(not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in stop_word_ids)
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )
        self.stop_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids))
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert len(stop_token_seq) > 0, "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: paddle.Tensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            return True
        elif len(tokens) > len(prev_tokens):
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    match = True
                    break
            stopped_samples.append(match)
        return stopped_samples
