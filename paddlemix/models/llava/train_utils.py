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


import copy
import re
from typing import Dict, Sequence

import paddle
from paddlenlp.transformers import PretrainedTokenizer

import paddlemix.models.llava.conversation as conversation_lib

from .constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)
from .mm_utils import tokenizer_image_token


def preprocess_multimodal(
    sources: Sequence[str],
    mm_use_im_start_end: bool = False,
) -> Dict:
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_llama_2(sources, tokenizer: PretrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    if has_image:
        input_ids = paddle.stack(
            x=[tokenizer_image_token(prompt, tokenizer, return_tensors="pd") for prompt in conversations], axis=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pd",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
    return dict(input_ids=input_ids, labels=targets)


def preprocess_v1(sources, tokenizer: PretrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []

    pattern_role_human = roles["human"]
    match_role_human = [m.start() for m in re.finditer(pattern_role_human, sources)]

    pattern_role_gpt = roles["gpt"]
    match_role_gpt = [n.start() for n in re.finditer(pattern_role_gpt, sources)]

    assert len(match_role_human) == len(match_role_gpt)
    conv.messages = []
    for i in range(len(match_role_human)):
        human_start = match_role_human[i]
        human_end = match_role_gpt[i]
        gpt_start = human_end
        gpt_end = match_role_human[i + 1] if i + 1 < len(match_role_human) else len(sources)
        query = sources[human_start + len(roles["human"]) : human_end]
        conv.append_message(conv.roles[0], query)
        ans = sources[gpt_start + len(roles["gpt"]) : gpt_end]
        conv.append_message(conv.roles[1], ans)
    conversations.append(conv.get_prompt())

    if has_image:
        input_ids = paddle.stack(
            x=[tokenizer_image_token(prompt, tokenizer, return_tensors="pd") for prompt in conversations], axis=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pd",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    sep = conv.sep + conv.roles[1] + ": "
    new_targets = []  # FIXME: In npu device, the inplace modification does not take effect
    for conversation, target in zip(conversations, targets):
        total_len = int(target.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
        new_targets.append(target)
    new_targets = paddle.stack(new_targets, axis=0)
    return dict(input_ids=input_ids, labels=new_targets)


def preprocess_plain(sources: Sequence[str], tokenizer: PretrainedTokenizer) -> Dict:
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0] = DEFAULT_IMAGE_TOKEN
        conversation = source[0] + source[1] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pd") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def get_conversation(
    version: Sequence[str], sources: Sequence[str], tokenizer: PretrainedTokenizer, has_image: bool = False
) -> Dict:
    """
        Given a list of sources, each is a conversation list. This transform:
        1. Add signal '### ' at the beginning each sentence, with end signal '
    ';
        2. Concatenate conversations together;
        3. Tokenize the concatenated conversation;
        4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    else:
        raise NotImplementedError
