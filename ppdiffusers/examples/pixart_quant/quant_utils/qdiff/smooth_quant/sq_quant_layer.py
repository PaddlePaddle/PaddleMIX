import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear  # 请确保这是 Paddle 版本的基类

class SQQuantizedLinear(QuantizedLinear):
    """
    Base quantized linear layer for Paddle,
    static weight quantization + dynamic activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: paddle.nn.Linear,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)

        self.alpha = quant_config.smooth_quant.alpha
        self.channel_mask = None  # 在 PTQ 阶段外部赋值

    def get_channel_mask(self, act_mask):
        """
        act_mask: 激活的通道级最大值（shape [C_in]）
        生成 channel_mask 并存到 self.channel_mask
        """
        # weight: [C_out, C_in]
        weight_abs = paddle.abs(self.fp_module.weight)
        # 在 axis=0（按行取最大）得到每列的最大值 -> shape [C_in]
        weight_mask = paddle.max(weight_abs, axis=1)
        # 避免负数**alpha 造成 nan（这里 weight_mask 和 act_mask 已取 abs）
        channel_mask = (paddle.abs(weight_mask) ** self.alpha) / (paddle.abs(act_mask) ** (1.0 - self.alpha))
        self.channel_mask = channel_mask

        # 检查 inf
        if paddle.isinf(self.channel_mask).any().item():
            raise AssertionError("inf exists in channel_mask")

    def update_quantized_weight_scaled(self):
        assert self.channel_mask is not None, "channel_mask is not set"
        C_in, C_out = self.fp_module.weight.shape

        # 关闭 w_quantizer 的 init 标志以重新计算
        self.w_quantizer.init_done = False

        # 对权重按通道缩放后量化
        scaled = self.fp_module.weight.t() / self.channel_mask.reshape([1, C_in])
        q_w = self.w_quantizer(scaled)

        # 在 Paddle 中推荐用 set_value 更新参数的值
        # q_w 必须是与 self.weight 形状相同的 Tensor
        self.weight.set_value(q_w.t())

        # 检查 nan
        if paddle.isnan(self.weight).any().item():
            raise AssertionError("nan exists in weight")

        self.w_quantizer.init_done = True

    def forward(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        """
        输入形状: [B, N_token, C]
        """
        if not getattr(self, "quant_mode", False):
            # 使用 FP module（注意 paddle 的 Linear 通常只接受输入）
            return self.fp_module(x)

        # quant 模式
        B, N_token, C = x.shape

        # 用 channel_mask 缩放激活
        x = x * self.channel_mask.reshape([1, 1, C])

        # 先展平以便 activation quantizer 工作（形状 [B*N_token, C]）
        x = x.reshape([B * N_token, -1])

        # 检查 nan
        if paddle.isnan(x).any().item():
            raise AssertionError("nan exists in x")

        # 激活量化（假设 a_quantizer 是兼容 Paddle 的 callable）
        x = self.a_quantizer(x)

        # 恢复形状
        x = x.reshape([B, N_token, C])

        # 使用量化（或 dequant 后）的 weight 做线性变换
        # Paddle 的 F.linear(input, weight, bias=None)
        y = F.linear(x, self.weight, self.bias)

        return y
