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

import paddle
from magvit2 import VideoTokenizer


def main():
    tokenizer = VideoTokenizer(
        image_size=128,
        init_dim=64,
        max_dim=512,
        codebook_size=1024,
        flash_attn=False,
        layers=(
            "residual",
            "compress_space",
            ("consecutive_residual", 2),
            "compress_space",
            ("consecutive_residual", 2),
            "linear_attend_space",
            "compress_space",
            ("consecutive_residual", 2),
            "attend_space",
            "compress_time",
            ("consecutive_residual", 2),
            "compress_time",
            ("consecutive_residual", 2),
            "attend_time",
        ),
    )

    video = paddle.randn([1, 3, 17, 128, 128])

    codes = tokenizer.tokenize(video)

    decoded_video = tokenizer.decode_from_code_indices(codes)

    print(decoded_video.shape)  # [1,3,17,128,128]


main()
