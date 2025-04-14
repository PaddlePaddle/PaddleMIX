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

import paddle
from paddlenlp.generation import LogitsProcessor


class SuppressTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = paddle.to_tensor(list(suppress_tokens))

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        vocab_tensor = paddle.arange(scores.shape[-1])
        suppress_token_mask = paddle.isin(vocab_tensor, self.suppress_tokens)
        scores_processed = paddle.where(suppress_token_mask, -float("inf"), scores)
        return scores_processed
