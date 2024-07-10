import paddle
import paddle.nn as nn


class RMSNorm(nn.Layer):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            out_6 = paddle.create_parameter(shape=paddle.ones(shape=dim).
                shape, dtype=paddle.ones(shape=dim).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.
                ones(shape=dim)))
            out_6.stop_gradient = not True
            self.weight = out_6

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (paddle.Tensor): The input tensor.

        Returns:
            paddle.Tensor: The normalized tensor.

        """
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (paddle.Tensor): The input tensor.

        Returns:
            paddle.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.astype(dtype='float32')).astype(dtype=x.dtype)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, dtype=None):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, dtype=dtype)

    def forward(self, x):
        y = super().forward(x).to(x.dtype)
        return y

def normalization(channels, dtype=None):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, dtype=dtype)
