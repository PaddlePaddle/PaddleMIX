import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer

from omegaconf import ListConfig



class QuantizedLinear(paddle.nn.Linear):
    """
    Paddle 版本的 QuantizedLinear
    - static weight quantization (w_quantizer)
    - dynamic activation quantization (a_quantizer)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: None = None,            # kept for API parity (ignored by Paddle here)
        quant_config: dict = None,
        fp_module: paddle.nn.Linear = None,
    ) -> None:
        # Paddle Linear: bias_attr expects bool or ParamAttr
        super().__init__(in_features, out_features, bias_attr=bias)

        self.fp_module = fp_module
        self.q_cfg = quant_config or {}

        # set default as None, to skip quant some part
        self.w_quantizer = None
        self.a_quantizer = None

        # weight quantizer init
        if self.q_cfg.get('weight', None) is not None:
            weight_cfg = self.q_cfg['weight']
            # detect ListConfig whether mixed-precision
            n_bits_attr = getattr(weight_cfg, 'n_bits', None)
            self.w_quantizer = StaticQuantizer(weight_cfg)

            # quantize the weight from FP module, bias remain as float
            if self.fp_module is None:
                raise ValueError("fp_module must be provided when weight quantization is enabled.")
            # assume fp_module.weight is a paddle.Tensor/Parameter
            # w_quantizer returns a tensor (dequantized) — set as this module's weight
            w_q = self.w_quantizer(self.fp_module.weight.t()).t()  # expected shape [out_features, in_features]
            # ensure shape matches
            if tuple(w_q.shape) != tuple(self.weight.shape):
                raise RuntimeError(f"quantized weight shape {w_q.shape} != target weight shape {tuple(self.weight.shape)}")
            # assign value to the Parameter
            self.weight.set_value(w_q)
            # mark quant init done (mirrors original behavior)
            self.w_quantizer.init_done = True
        else:
            # copy fp weight to this layer
            if self.fp_module is None:
                # keep default random init if no fp_module provided
                pass
            else:
                if tuple(self.fp_module.weight.shape) != tuple(self.weight.shape):
                    raise RuntimeError("fp_module.weight shape mismatch")
                self.weight.set_value(self.fp_module.weight)

        # save references to fp weight and bias
        self.fp_weight = self.fp_module.weight if self.fp_module is not None else None
        # assign bias value from fp_module if provided and bias exists
        if self.fp_module is not None and self.fp_module.bias is not None:
            # paddle Linear 'bias' is a Parameter if bias_attr True
            if self.bias is not None:
                self.bias.set_value(self.fp_module.bias)
        # else keep default

        # activation quantizer init
        if self.q_cfg.get('act', None) is not None:
            act_cfg = self.q_cfg['act']
            act_n_bits = getattr(act_cfg, 'n_bits', None)
            self.a_quantizer = DynamicQuantizer(act_cfg)

        self.use_kernel = False  # whether use the cuda kernel for actual saving (same flag name)
        self.quant_mode = True   # when set as False, use the original model forward

    def forward(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        """
        input shape: [B, N_token, C] (C == in_features)
        behavior:
            if not self.quant_mode: call fp_module (if provided)
            else: reshape to [B*N_token, -1], quantize activation (if any),
                  reshape back and apply linear with quantized weight/bias
        """
        if not self.quant_mode:
            # use the FP module if provided
            if self.fp_module is not None:
                return self.fp_module(x, *args, **kwargs)
            else:
                # fallback to parent Linear forward
                return super().forward(x)
        else:
            # ensure x has three dims
            if x.ndim != 3:
                raise ValueError("Expected x shape [B, N_token, C] for QuantizedLinear forward.")
            B, N_token, C = x.shape
            # reshape to [B*N_token, C]
            x_reshaped = paddle.reshape(x, [B * N_token, -1])

            # quantize activation if present (DynamicQuantizer expects [G, -1] in original design)
            if self.a_quantizer is not None:
                x_q = self.a_quantizer(x_reshaped)
            else:
                x_q = x_reshaped

            # reshape back to [B, N_token, C]
            x_back = paddle.reshape(x_q, [B, N_token, C])

            # perform linear using stored (quantized) weight and bias
            # paddle.nn.functional.linear works like torch.nn.functional.linear
            y = F.linear(x_back, self.weight, self.bias)
            return y


if __name__ == "__main__":
    # quick test for the Paddle QuantizedLinear
    paddle.set_device('gpu')  # 改成 'gpu' 如果你要在 GPU 上测试

    # prepare a FP linear as fp_module
    in_features = 16
    out_features = 8
    fp_linear = paddle.nn.Linear(in_features, out_features, bias_attr=True)
    # init deterministic for testing
    paddle.seed(42)

    # simple quant_config example (symmetric 8-bit for both weight and act)
    quant_config = {
        'weight': {
            'n_bits': 8,
            'sym': True
        },
        'act': {
            'n_bits': 8,
            'sym': True
        }
    }

    # NOTE: If your quantizers expect OmegaConf nodes, wrap dict into an object/Namespace or adjust above logic.
    # For this test we'll assume dict works and quantizer constructors accept such dict-like config.

    # create QuantizedLinear (Paddle)
    qlinear = QuantizedLinear(in_features, out_features, bias=True, device=None, quant_config=quant_config, fp_module=fp_linear)

    # build a dummy input [B, N_token, C]
    B = 2
    N_token = 3
    C = in_features
    x = paddle.randn([B, N_token, C], dtype='float32')

    # forward in quant mode
    qlinear.quant_mode = True
    y_q = qlinear(x)
    print("Quant mode output shape:", y_q.shape)
    print("Sample output (quant):", y_q.flatten()[:6].numpy())

    # forward in FP mode (use fp_module)
    qlinear.quant_mode = False
    y_fp = qlinear(x)
    print("FP mode output shape:", y_fp.shape)
    print("Sample output (fp):", y_fp.flatten()[:6].numpy())

    print("Done.")