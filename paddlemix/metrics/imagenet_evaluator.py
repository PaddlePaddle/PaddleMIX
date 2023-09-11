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

import paddle
from tqdm import tqdm

from .clip_zero_shot import accuracy, get_autocast, get_cast_dtype


class ImageNetEvaluator:
    def __init__(self, args):
        self.batch_size = args.per_device_eval_batch_size
        self.cast_dtype = get_cast_dtype(args)

    def clas_eval(self, evalres):
        results = {}
        predictions, labels = evalres.predictions, evalres.label_ids
        N = predictions.shape[0]
        top1, top5 = 0.0, 0.0

        autocast = get_autocast(self.cast_dtype)
        with paddle.no_grad():
            for step in tqdm(range((N + self.batch_size - 1) // self.batch_size)):
                with autocast():
                    logits = paddle.to_tensor(predictions[step * self.batch_size : (step + 1) * self.batch_size])
                    target = paddle.to_tensor(labels[step * self.batch_size : (step + 1) * self.batch_size])
                if logits.shape[-1] < 5:
                    (acc1,) = accuracy(logits, target, topk=(1,))
                    acc5 = -1
                else:
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5

        top1 = top1 / N
        top5 = top5 / N
        results["top1"] = top1
        results["top5"] = top5
        print("Finished evaluation.")
        return results
