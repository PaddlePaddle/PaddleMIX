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

import math
from functools import partial

import paddle


def log(t, eps=1e-20):
    return paddle.log(x=t.clip(min=eps))


def gumbel_noise(t, generator=None):
    noise = paddle.zeros_like(x=t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, generator=None):
    return (t / max(temperature, 1e-10) + gumbel_noise(t, generator=generator)).argmax(axis=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * tuple(logits.shape)[-1])
    val, ind = logits.topk(k=k, axis=-1)
    probs = paddle.full_like(x=logits, fill_value=float("-inf"))
    probs.put_along_axis_(axis=2, indices=ind, values=val, broadcast=False)
    return probs


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    # confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    confidence = log(probs) + temperature * gumbel_noise(probs)
    sorted_confidence = paddle.sort(confidence, axis=-1)
    cut_off = paddle.take_along_axis(
        sorted_confidence, axis=1, indices=mask_len.astype(dtype="int64"), broadcast=False
    )
    masking = confidence < cut_off
    return masking


def cosine_schedule(t):
    return paddle.cos(x=t * math.pi * 0.5)


def linear_schedule(t):
    mask_ratio = 1 - t
    mask_ratio = mask_ratio.clip(min=1e-06, max=1.0)
    return mask_ratio


def pow(t, method):
    exponent = float(method.replace("pow", ""))
    mask_ratio = 1.0 - t**exponent
    mask_ratio = mask_ratio.clip(min=1e-06, max=1.0)
    return mask_ratio


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-06):
    for item in [t, start, end, tau]:
        item = paddle.to_tensor(data=item) if not paddle.is_tensor(x=item) else item
    v_start = paddle.nn.functional.sigmoid(x=paddle.to_tensor(data=start / tau))
    v_end = paddle.nn.functional.sigmoid(x=paddle.to_tensor(data=end / tau))
    output = paddle.nn.functional.sigmoid(x=(t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return paddle.clip(x=output, min=clip_min, max=1.0)


def get_mask_chedule(method, **schedule_kwargs):
    if method == "cosine":
        return cosine_schedule
    elif method == "linear":
        return linear_schedule
    elif "pow" in method:
        return partial(pow, method=method)
    elif method == "sigmoid":
        return partial(sigmoid_schedule, **schedule_kwargs)
    else:
        raise ValueError("Unknown schedule method: {}".format(method))


def top_k_top_p_filtering(
    logits: paddle.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> paddle.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])
        indices_to_remove = logits < paddle.topk(k=top_k, x=logits)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = paddle.sort(descending=True, x=logits), paddle.argsort(
            descending=True, x=logits
        )
        cumulative_probs = paddle.cumsum(x=paddle.nn.functional.softmax(x=sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.put_along_axis(
            axis=1, indices=sorted_indices, values=sorted_indices_to_remove, broadcast=False
        )
        logits[indices_to_remove] = filter_value
    return logits
