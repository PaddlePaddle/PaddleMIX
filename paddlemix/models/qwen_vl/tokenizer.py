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

import base64
import os
import unicodedata
from typing import Any, Callable, Collection, Dict, List, Set, Tuple, Union

from paddlemix.utils.log import logger

try:
    import tiktoken
except:
    logger.warning("tiktoken not import, if you want to use tiktoken, require python>=3.8 and pip install tiktoken")
    pass

from paddlenlp.transformers import AddedToken, PretrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "qwen.tiktoken", "ttf": "SimSun.ttf"}
PAT_STR = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
EXTRAS = tuple(f"<|extra_{i}|>" for i in range(205))
SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND) + EXTRAS
IMG_TOKEN_SPAN = 256
FONT_PATH = "SimSun.ttf"


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)
    }


def _list_find(input_list: List[Any], candidates: Tuple[Any], start: int = 0):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1


def _replace_closed_tag(
    input_tokens: List[Any],
    start_tags: Union[Any, Tuple[Any]],
    end_tags: Union[Any, Tuple[Any]],
    inclusive_replace_func: Callable,
    exclusive_replace_func: Callable = lambda x: x,
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)
    output_tokens = []
    end = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end:start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed image token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start : end + 1]))
        end += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))
    return output_tokens


class QWenVLTokenizer(PretrainedTokenizer):
    """QWen tokenizer."""

    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    resource_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors="replace",
        image_start_tag="<img>",
        image_end_tag="</img>",
        image_pad_tag="<imgpad>",
        ref_start_tag="<ref>",
        ref_end_tag="</ref>",
        box_start_tag="<box>",
        box_end_tag="</box>",
        quad_start_tag="<quad>",
        quad_end_tag="</quad>",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.image_pad_tag = image_pad_tag
        self.ref_start_tag = ref_start_tag
        self.ref_end_tag = ref_end_tag
        self.box_start_tag = box_start_tag
        self.box_end_tag = box_end_tag
        self.quad_start_tag = quad_start_tag
        self.quad_end_tag = quad_end_tag
        self.IMAGE_ST = (
            ref_start_tag,
            ref_end_tag,
            box_start_tag,
            box_end_tag,
            quad_start_tag,
            quad_end_tag,
            image_start_tag,
            image_end_tag,
            image_pad_tag,
        )
        self.errors = errors
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)
        self.special_tokens = {
            token: index for index, token in enumerate(SPECIAL_TOKENS + self.IMAGE_ST, start=len(self.mergeable_ranks))
        }
        self.img_start_id = self.special_tokens[self.image_start_tag]
        self.img_end_id = self.special_tokens[self.image_end_tag]
        self.img_pad_id = self.special_tokens[self.image_pad_tag]
        self.ref_start_id = self.special_tokens[self.ref_start_tag]
        self.ref_end_id = self.special_tokens[self.ref_end_tag]
        self.box_start_id = self.special_tokens[self.box_start_tag]
        self.box_end_id = self.special_tokens[self.box_end_tag]
        self.quad_start_id = self.special_tokens[self.quad_start_tag]
        self.quad_end_id = self.special_tokens[self.quad_end_tag]
        enc = tiktoken.Encoding(
            "Qwen", pat_str=PAT_STR, mergeable_ranks=self.mergeable_ranks, special_tokens=self.special_tokens
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"
        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
        self.tokenizer = enc
        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "Qwen", pat_str=PAT_STR, mergeable_ranks=self.mergeable_ranks, special_tokens=self.special_tokens
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return self.mergeable_ranks

    def convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes, str]]]) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _update_tiktoken(self, tokens: List[str], special_tokens: bool = False) -> int:
        if special_tokens:
            added_tokens = []
            for token in tokens:
                if token in self.special_tokens:
                    continue

                token_id = len(self.mergeable_ranks) + len(self.special_tokens)
                self.special_tokens[token] = token_id
                self.decoder[token_id] = token

                added_tokens.append(token)

            import tiktoken

            self.tokenizer = tiktoken.Encoding(
                "Qwen",
                pat_str=PAT_STR,
                mergeable_ranks=self.mergeable_ranks,
                special_tokens=self.special_tokens,
            )

            return len(added_tokens)
        else:
            raise ValueError("Adding regular tokens is not supported")

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")
        new_tokens_str = []
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            new_tokens_str.append(surface_form)

        return self._update_tiktoken(new_tokens_str, special_tokens)

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "qwen.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])

        def _encode_imgurl(img_tokens):
            assert img_tokens[0] == self.image_start_tag and img_tokens[-1] == self.image_end_tag
            img_tokens = img_tokens[1:-1]
            img_url = b"".join(img_tokens)
            out_img_tokens = list(map(self.decoder.get, img_url))
            if len(out_img_tokens) > IMG_TOKEN_SPAN:
                logger.warning(
                    "The content in {}..{} is too long,will use [self.image_pad_tag] * IMG_TOKEN_SPAN replace. make sure use QwenVLProcessor for get input data".format(
                        self.image_start_tag, self.image_end_tag
                    )
                )
                out_img_tokens = [self.image_pad_tag] * IMG_TOKEN_SPAN
            else:
                out_img_tokens.extend([self.image_pad_tag] * (IMG_TOKEN_SPAN - len(out_img_tokens)))

            out_img_tokens = [self.image_start_tag] + out_img_tokens + [self.image_end_tag]
            return out_img_tokens

        return _replace_closed_tag(tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl)

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False, errors: str = None, **kwargs
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        def _decode_imgurl(img_token_ids):
            assert img_token_ids[0] == self.img_start_id and img_token_ids[-1] == self.img_end_id
            img_token_ids = img_token_ids[1:-1]
            img_token_ids = img_token_ids[: img_token_ids.index(self.img_pad_id)]
            img_url = bytes(img_token_ids).decode("utf-8")
            return [self.img_start_id] + self.tokenizer.encode(img_url) + [self.img_end_id]

        token_ids = _replace_closed_tag(token_ids, self.img_start_id, self.img_end_id, _decode_imgurl)
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def to_list_format(self, text: str):
        text = unicodedata.normalize("NFC", text)
        token_ids = self.tokenizer.encode(text, allowed_special=set(self.IMAGE_ST + (ENDOFTEXT,)))

        def _encode_vl_info(tokens):
            if len(tokens) == 0:
                return []
            if tokens[0] == self.img_start_id and tokens[-1] == self.img_end_id:
                key = "image"
            elif tokens[0] == self.ref_start_id and tokens[-1] == self.ref_end_id:
                key = "ref"
            elif tokens[0] == self.box_start_id and tokens[-1] == self.box_end_id:
                key = "box"
            elif tokens[0] == self.quad_start_id and tokens[-1] == self.quad_end_id:
                key = "quad"
            else:
                _tobytes = lambda x: x.encode("utf-8") if isinstance(x, str) else x
                return [{"text": b"".join(map(_tobytes, map(self.decoder.get, tokens))).decode("utf-8")}]
            _tobytes = lambda x: x.encode("utf-8") if isinstance(x, str) else x
            val = b"".join(map(_tobytes, map(self.decoder.get, tokens[1:-1]))).decode("utf-8")
            return [{key: val}]

        return _replace_closed_tag(
            token_ids,
            (self.img_start_id, self.ref_start_id, self.box_start_id, self.quad_start_id),
            (self.img_end_id, self.ref_end_id, self.box_end_id, self.quad_end_id),
            _encode_vl_info,
            _encode_vl_info,
        )

    def from_list_format(self, list_format: List[Dict]):
        text = ""
        num_images = 0
        for ele in list_format:
            if "image" in ele:
                num_images += 1
                text += f"Picture {num_images}:"
                text += self.image_start_tag + ele["image"] + self.image_end_tag
                text += "\n"
            elif "text" in ele:
                text += ele["text"]
            elif "box" in ele:
                if "ref" in ele:
                    text += self.ref_start_tag + ele["ref"] + self.ref_end_tag
                for box in ele["box"]:
                    text += (
                        self.box_start_tag + "(%d,%d),(%d,%d)" % (box[0], box[1], box[2], box[3]) + self.box_end_tag
                    )
            else:
                raise ValueError("Unsupported element: " + str(ele))
        return text

    def _fetch_latest_picture(self, response, history):
        if history is None:
            history = []
        _history = history + [(response, None)]
        for q, r in _history[::-1]:
            for ele in self.to_list_format(q)[::-1]:
                if "image" in ele:
                    return ele["image"]
        return None

    def _fetch_all_box_with_ref(self, text):
        list_format = self.to_list_format(text)
        output = []
        for i, ele in enumerate(list_format):
            if "box" in ele:
                bbox = tuple(map(int, ele["box"].replace("(", "").replace(")", "").split(",")))
                assert len(bbox) == 4
                output.append({"box": bbox})
                if i > 0 and "ref" in list_format[i - 1]:
                    output[-1]["ref"] = list_format[i - 1]["ref"].strip()
        return output
