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


def dense_connector_dci(image_features, image_forward_outs, is_siglip=True):
    image_features_1 = []
    image_features_2 = []
    if not is_siglip:
        for i in range(0, 12):
            image_features_1.append(image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype))
        image_features_1 = paddle.stack(image_features_1, axis=0)
        image_features_1 = paddle.sum(image_features_1, axis=0) / 12
        for i in range(12, 24):
            image_features_2.append(image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype))
        image_features_2 = paddle.stack(image_features_2, axis=0)
        image_features_2 = paddle.sum(image_features_2, axis=0) / 12
    else:
        for i in range(0, 13):
            image_features_1.append(image_forward_outs.hidden_states[i][:, :].to(image_features.dtype))
        image_features_1 = paddle.stack(image_features_1, axis=0)
        image_features_1 = paddle.sum(image_features_1, axis=0) / 13
        for i in range(13, 26):
            image_features_2.append(image_forward_outs.hidden_states[i][:, :].to(image_features.dtype))
        image_features_2 = paddle.stack(image_features_2, axis=0)
        image_features_2 = paddle.sum(image_features_2, axis=0) / 13
    return paddle.concat([image_features_1, image_features_2], axis=-1)


def dense_connector(image_features, image_forward_outs, is_siglip=True, mm_dense_connector_type="dci"):
    if mm_dense_connector_type == "dci":
        image_features_dc = dense_connector_dci(image_features, image_forward_outs, is_siglip)
        image_features = paddle.concat((image_features, image_features_dc), axis=-1)
    else:
        raise NotImplementedError()

    return image_features
