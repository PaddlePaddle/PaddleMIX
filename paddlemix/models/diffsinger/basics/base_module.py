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


class CategorizedModule(paddle.nn.Layer):
    @property
    def category(self):
        raise NotImplementedError()

    def check_category(self, category):
        if category is None:
            raise RuntimeError(
                """Category is not specified in this checkpoint.
If this is a checkpoint in the old format, please consider migrating it to the new format via the following command:
python scripts/migrate.py ckpt <INPUT_CKPT> <OUTPUT_CKPT>"""
            )
        elif category != self.category:
            raise RuntimeError(
                f"""Category mismatches!
This checkpoint is of the category '{category}', but a checkpoint of category '{self.category}' is required."""
            )
