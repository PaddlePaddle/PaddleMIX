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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paddle


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: paddle.dtype,
        device: Union[str, "str"] = "cpu",
    ) -> Optional[paddle.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")
        input_shape = batch_size, query_length
        past_key_values_length = key_value_length - query_length
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: paddle.Tensor,
        query_length: int,
        dtype: paddle.dtype,
        key_value_length: Optional[int] = None,
    ) -> paddle.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = tuple(attention_mask_2d.shape)[0], query_length
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )
            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.place,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.place
        )
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(
                mask=expanded_attn_mask.astype(dtype="bool"), value=paddle.finfo(dtype=dtype).min
            )
        expanded_4d_mask = expanded_attn_mask
        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: list,
        dtype: paddle.dtype,
        device: str,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=paddle.finfo(dtype=dtype).min)
        mask_cond = paddle.arange(end=mask.shape[-1])
        mask.masked_fill_(mask=mask_cond < (mask_cond + 1).view([mask.shape[-1], 1]), value=0)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = paddle.concat(x=[paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = paddle.tril(x=paddle.ones_like(x=mask, dtype="bool"), diagonal=diagonal)
            mask.masked_fill_(mask=context_mask, value=paddle.finfo(dtype=dtype).min)
        return mask[None, None, :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])

    @staticmethod
    def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = tuple(mask.shape)
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(shape=[bsz, 1, tgt_len, src_len]).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(mask=inverted_mask, value=paddle.finfo(dtype=dtype).min)

    @staticmethod
    def _unmask_unattended(expanded_mask: paddle.Tensor, min_dtype: float):
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        if expanded_mask.dtype == "bool":
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )
        return expanded_mask.mul(~paddle.all(x=expanded_mask == min_dtype, axis=-1, keepdim=True))

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: Optional[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        is_training: bool = False,
    ) -> bool:
        """
        Detects whether the optional user-specified attention_mask & the automatically created causal mask can be
        ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

        In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
        `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
        allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
        passed).
        """
        _, query_length = tuple(inputs_embeds.shape)[0], tuple(inputs_embeds.shape)[1]
        key_value_length = query_length + past_key_values_length
        ignore_causal_mask = False
        if attention_mask is None:
            if (
                is_training
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(tuple(attention_mask.shape)) == 4:
                return False
            elif paddle.all(x=attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    ignore_causal_mask = True
        return ignore_causal_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[paddle.Tensor],
    input_shape: Union[list, Tuple, List],
    inputs_embeds: paddle.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    if attention_mask is not None and len(tuple(attention_mask.shape)) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(tuple(attention_mask.shape)) == 4:
        expected_shape = input_shape[0], 1, input_shape[1], key_value_length
        if tuple(tuple(attention_mask.shape)) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(tuple(attention_mask.shape))}; expected: {expected_shape}."
            )
        else:
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                mask=inverted_mask.to("bool"), value=paddle.finfo(dtype=inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.place
        )
    return attention_mask
