import logging
import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union
import time
import math
from omegaconf import ListConfig

logger = logging.getLogger(__name__)

class BaseQuantizer(nn.Layer):

    def __init__(self, quant_config):
        super(BaseQuantizer, self).__init__()
        
        # unpack the quant configurations
        self.n_bits = quant_config['n_bits']
        # self.group = quant_config['group']
        self.sym = quant_config.get('sym', False)

        if isinstance(self.n_bits, list):
            raise AssertionError("when multiple n_bits are adopted, use the MixedPrecisionBaseQuantizer")
        # assert self.group in ['token','tensor','channel']

        # Paddle doesn't require register_buffer like PyTorch for simple use:
        # we'll keep them as attributes and set them to paddle.Tensor when available.
        self.delta = None
        self.zero_point = None

        # INFO: for mixed_precision, the n_bits could be a ListConfig, and need to be initialized in subclass init
        if not isinstance(self.n_bits, ListConfig):
            # note: for symmetric case we use slightly different formula same as your original
            self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1

        self.init_done = False
        # module_name used in logger messages in original; keep it to avoid attribute error
        self.module_name = self.__class__.__name__

    def forward(self, x: paddle.Tensor):
        raise NotImplementedError("should be implemented in subclass.")
    
    def init_quant_params(self, x):
        raise NotImplementedError("should be implemented in subclass.")


class StaticQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

        if self.sym:
            self.x_absmax = None
        else:
            self.x_max = None
            self.x_min = None
    
    def forward(self, x: paddle.Tensor):
        x_quant = self.quantize(x)
        # ensure delta and zero_point are tensors
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant
    
    def quantize(self, x: paddle.Tensor):
    
        if self.init_done is not True:  # set as True externally when done
            self.init_quant_params(x)
        # x_int = round(x / delta) - zero_point
        x_int = paddle.round(x / self.delta) - self.zero_point
        # clamp: note paddle.clip takes min and max scalars or tensors broadcastable
        x_quant = paddle.clip(x_int, min=-self.n_levels - 1, max=self.n_levels)
        return x_quant
    
    def init_quant_params(self, x: paddle.Tensor):

        assert len(x.shape) == 2  # [N_group, -1]
        if self.sym:
            # x_absmax per group
            x_absmax = paddle.max(paddle.abs(x), axis=1)
            # update stored x_absmax
            if self.x_absmax is not None:
                try:
                    self.x_absmax = paddle.maximum(self.x_absmax, x_absmax)
                except Exception:
                    # if devices differ, user should ensure consistent device or convert
                    self.x_absmax = paddle.maximum(self.x_absmax, x_absmax)
            else:
                self.x_absmax = x_absmax
            delta = x_absmax / self.n_levels
            zero_point = paddle.zeros_like(delta)
        else:
            x_max = paddle.max(x, axis=1)
            # set negative maxima to 0
            x_max = paddle.where(x_max < 0., paddle.zeros_like(x_max), x_max)

            if self.x_max is not None:
                try:
                    self.x_max = paddle.maximum(self.x_max, x_max)
                except Exception:
                    # device mismatch handling: convert if necessary (user may need to ensure devices)
                    self.x_max = paddle.maximum(self.x_max, x_max)
            else:
                self.x_max = x_max

            x_min = paddle.min(x, axis=1)
            x_min = paddle.where(x_min > 0., paddle.zeros_like(x_min), x_min)
            if self.x_min is not None:
                try:
                    self.x_min = paddle.minimum(self.x_min, x_min)
                except Exception:
                    self.x_min = paddle.minimum(self.x_min, x_min)
            else:
                self.x_min = x_min

            delta = (x_max - x_min) / (self.n_levels - 1)
            # zero_point formula preserved
            zero_point = paddle.round(x_min / delta) + (self.n_levels / 2)
        
        try:
            # use paddle.logical_and to check > eps for all elements
            assert bool(paddle.all(delta > 1.e-6))
        except Exception as e:
            # drop into debugger equivalently, here we just raise for visibility
            # If you want interactive debugging, you can import ipdb and set_trace here like original.
            raise AssertionError("unexpected small delta exists") from e

        # unsqueeze last dim
        self.delta = paddle.unsqueeze(delta, axis=-1)  # [G] -> [G,1]
        self.zero_point = paddle.unsqueeze(zero_point, axis=-1)


class DynamicQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    compute quant params on-the-fly
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def quantize(self, x: paddle.Tensor):
         # get the quant_params online
        assert len(x.shape) == 2  # [N_group, -1]
        assert int(paddle.sum(paddle.isnan(x)).numpy()) == 0  # no nan exists

        if self.sym:
            x_absmax = paddle.max(paddle.abs(x), axis=1)
            self.x_absmax = x_absmax
            
            delta = x_absmax / self.n_levels
            zero_point = paddle.zeros_like(delta)
            
            eps = 1.e-6
            try:
                assert bool(paddle.all(paddle.abs(delta) > eps))
            except Exception:
                # fallback: set small delta to eps
                delta = paddle.where(paddle.abs(delta) < eps, paddle.full_like(delta, eps), delta)
                logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(float(paddle.min(paddle.abs(delta)).numpy()), self.module_name))
                
        else:
            x_max = paddle.max(x, axis=1)
            x_max = paddle.where(x_max < 0., paddle.zeros_like(x_max), x_max)
            self.x_max = x_max

            x_min = paddle.min(x, axis=1)
            x_min = paddle.where(x_min > 0., paddle.zeros_like(x_min), x_min)
            self.x_min = x_min

            delta = (x_max - x_min) / (self.n_levels - 1)
            # INFO: check small values for delta
            eps = 1.e-8
            try:
                assert bool(paddle.all(paddle.abs(delta) > eps))
            except Exception:
                # fallback: set values smaller than eps to eps
                delta = paddle.where(paddle.abs(delta) < eps, paddle.full_like(delta, eps), delta)
                logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(float(paddle.min(paddle.abs(delta)).numpy()), self.module_name))
            zero_point = paddle.round(x_min / delta) + (self.n_levels / 2)

        self.delta = paddle.unsqueeze(delta, axis=-1)  # [G] -> [G,1]
        self.zero_point = paddle.unsqueeze(zero_point, axis=-1)

        # quantize model with quant params
        x_int = paddle.round(x / self.delta) - self.zero_point
        x_quant = paddle.clip(x_int, min=-self.n_levels - 1, max=self.n_levels)
        return x_quant

    def forward(self, x: paddle.Tensor):
        x_quant = self.quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant
    
if __name__ == '__main__':
    paddle.set_device('cpu')  # 你也可以改成 'gpu' 测试

    # 构造一个量化配置
    quant_config = {
        'n_bits': 8,
        'sym': True
    }

    # 测试 StaticQuantizer
    print("==== StaticQuantizer Test ====")
    static_quantizer = StaticQuantizer(quant_config)

    # 构造一个 [Group, Feature] 的张量，比如 4 组，每组 16 个元素
    x = paddle.randn([4, 16], dtype='float32')

    # 第一次 forward 时会调用 init_quant_params
    y_static = static_quantizer(x)
    print("Input shape:", x.shape)
    print("Output shape (Static):", y_static.shape)
    print("Delta shape:", static_quantizer.delta.shape)
    print("Zero point shape:", static_quantizer.zero_point.shape)
    print("Sample output (Static):", y_static[0, :5])

    # 测试 DynamicQuantizer
    print("\n==== DynamicQuantizer Test ====")
    dynamic_quantizer = DynamicQuantizer(quant_config)

    y_dynamic = dynamic_quantizer(x)
    print("Input shape:", x.shape)
    print("Output shape (Dynamic):", y_dynamic.shape)
    print("Delta shape:", dynamic_quantizer.delta.shape)
    print("Zero point shape:", dynamic_quantizer.zero_point.shape)
    print("Sample output (Dynamic):", y_dynamic[0, :5])

    print("\n✅ Paddle Quantizer forward 测试完成！")
    print(x[0,:5])
