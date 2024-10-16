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

from datetime import datetime

from magvit2 import VideoTokenizer
from trainer import VideoTokenizerTrainer

RUNTIME = datetime.now().strftime("%y%m%d %H:%M:%S")

tokenizer = VideoTokenizer(
    image_size=128,
    init_dim=64,
    max_dim=512,
    use_gan=False,
    use_fsq=False,
    codebook_size=2**18,
    perceptual_loss_weight=0,
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

trainer = VideoTokenizerTrainer(
    tokenizer,
    dataset_folder="/paddle/data/coco/images/val2014",
    dataset_type="images",  # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size=1,
    valid_frac=0,
    grad_accum_every=1,
    num_train_steps=100,
    max_grad_norm=1.0,
    learning_rate=1e-3,
    random_split_seed=85,
    optimizer_kwargs={"betas": (0.9, 0.99)},  # From the paper
    ema_kwargs={},
)

trainer.train()
