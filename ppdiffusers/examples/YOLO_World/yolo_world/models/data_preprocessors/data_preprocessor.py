#!/usr/bin/env python3
import paddle
import paddle.nn as nn

import numpy as np

class YOLOWDetDataPreprocessor(nn.Layer):
    def __init__(self,
                 mean=[0., 0., 0.],
                 std=[255., 255., 255.],
                 channel_conversion=True,
                 need_normalize=True):
        super().__init__()
        self.mean = paddle.Tensor(np.array(mean)).reshape([-1, 1, 1])
        self.std = paddle.Tensor(np.array(std)).reshape([-1, 1, 1])
        self.channel_conversion = channel_conversion
        self.need_normalize = need_normalize

    def cast_data(self, data):
        """Copying data to the target device."""
        if isinstance(data, dict):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, (list, tuple)):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, paddle.Tensor):
            return paddle.to_tensor(data, place=self.device, dtype=data.dtype)
        else:
            return data


    def forward(self, data, training=False):
        if not training:
            data = self.cast_data(data)
            _batch_input = data["image"]
            if self.channel_conversion:
                _batch_input = _batch_input[:, [2, 1, 0], ...]
            _batch_input = _batch_input.astype("float32")
            if self._enable_normalize:
                _batch_input = (_batch_input - self.mean) / self.std
            data["image"] = _batch_input
            return {"image": _batch_input, "text": data["text"]}
