import copy
import random
from typing import Optional, Tuple

import os
import re
import numpy as np

import paddle

import torch
import torch.nn as nn



# # 这部分用于对齐 torch 和 paddle 的 MultiheadAttention
# if __name__ == "__main__":

#     np.random.seed(1107)

#     x = np.random.randn(32, 255, 768) * 10000
#     x = x.astype("float32")

#     x_paddle = paddle.to_tensor(x)
#     x_torch = torch.from_numpy(x).cuda()

#     n_heads = 12
#     d_model = 768
#     dropout = 0

#     # -------- torch --------
#     self_attn_torch = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).cuda()
#     # -------- paddle --------
#     self_attn_paddle = paddle.nn.MultiHeadAttention(d_model, n_heads, dropout=dropout)


#     # -----------------------------
#     attn_torch_state_dict = self_attn_torch.state_dict()
#     attn_paddle_state_dict = self_attn_paddle.state_dict()

#     en_param_torch = set(attn_torch_state_dict.keys())
#     en_param_paddle = set(attn_paddle_state_dict.keys())

#     # torch 参数传给 Paddle
#     for key_torch, value_torch in attn_torch_state_dict.items():

#         # print(key_torch)

#         if 'out_proj.weight' == key_torch:
#             attn_paddle_state_dict[key_torch] = paddle.to_tensor(value_torch.cpu().numpy()).T
#             continue

#         if key_torch in en_param_paddle:
#             assert attn_paddle_state_dict[key_torch].shape == list(value_torch.shape)
#             attn_paddle_state_dict[key_torch] = paddle.to_tensor(value_torch.cpu().numpy())
#             continue

#         print(f"{key_torch} -> sth happened")


#     # 参数矩阵
#     q, k, v = torch.chunk(attn_torch_state_dict['in_proj_weight'], 3)
#     some_deal = lambda x: paddle.to_tensor(x.cpu().numpy()).T  # <-------- 转置?
#     q, k, v = some_deal(q), some_deal(k), some_deal(v)

#     attn_paddle_state_dict['q_proj.weight'] = q
#     attn_paddle_state_dict['k_proj.weight'] = k
#     attn_paddle_state_dict['v_proj.weight'] = v

#     # 偏置
#     q, k, v = torch.chunk(attn_torch_state_dict['in_proj_bias'], 3)
#     some_deal = lambda x: paddle.to_tensor(x.cpu().numpy())
#     q, k, v = some_deal(q), some_deal(k), some_deal(v)

#     attn_paddle_state_dict['q_proj.bias'] = q
#     attn_paddle_state_dict['k_proj.bias'] = k
#     attn_paddle_state_dict['v_proj.bias'] = v


#     # 加载参数
#     self_attn_paddle.load_dict(attn_paddle_state_dict)

#     # torch
#     tgt2_torch = self_attn_torch(x_torch, x_torch, x_torch)[0]

#     # paddle
#     tgt2_paddle = self_attn_paddle(x_paddle, x_paddle, x_paddle)

#     print(  
#         tgt2_torch.mean().item() - tgt2_paddle.mean().item(),
#         tgt2_torch.std().item()  - tgt2_paddle.std().item(),
#     )


def MultiheadAttention_torch2paddle(pd_model, tc_model, prefix=""):

    self_attn_torch, self_attn_paddle = tc_model, pd_model

    # 多头注意力机制，torch 模型转 paddle 模型
    attn_torch_state_dict = self_attn_torch.state_dict()
    attn_paddle_state_dict = self_attn_paddle.state_dict()

    en_param_torch = set(attn_torch_state_dict.keys())
    en_param_paddle = set(attn_paddle_state_dict.keys())

    # torch 参数传给 Paddle
    for key_torch, value_torch in attn_torch_state_dict.items():

        # print(key_torch)

        if prefix + 'out_proj.weight' == key_torch:
            # Linear 层参数要转置
            attn_paddle_state_dict[key_torch] = paddle.to_tensor(value_torch.cpu().numpy()).T
            continue

        if key_torch in en_param_paddle:
            assert attn_paddle_state_dict[key_torch].shape == list(value_torch.shape)
            attn_paddle_state_dict[key_torch] = paddle.to_tensor(value_torch.cpu().numpy())
            continue

        print(f"{key_torch} -> sth happened")


    # 参数矩阵
    q, k, v = torch.chunk(attn_torch_state_dict[prefix + 'in_proj_weight'], 3)
    some_deal = lambda x: paddle.to_tensor(x.cpu().numpy()).T  # <-------- 转置?
    q, k, v = some_deal(q), some_deal(k), some_deal(v)

    attn_paddle_state_dict[prefix + 'q_proj.weight'] = q
    attn_paddle_state_dict[prefix + 'k_proj.weight'] = k
    attn_paddle_state_dict[prefix + 'v_proj.weight'] = v

    # 偏置
    q, k, v = torch.chunk(attn_torch_state_dict[prefix + 'in_proj_bias'], 3)
    some_deal = lambda x: paddle.to_tensor(x.cpu().numpy())
    q, k, v = some_deal(q), some_deal(k), some_deal(v)

    attn_paddle_state_dict[prefix + 'q_proj.bias'] = q
    attn_paddle_state_dict[prefix + 'k_proj.bias'] = k
    attn_paddle_state_dict[prefix + 'v_proj.bias'] = v


    # 加载参数
    self_attn_paddle.load_dict(attn_paddle_state_dict)

    return self_attn_paddle


# # 测试 MultiheadAttention_torch2paddle 函数
# if __name__ == "__main__":

#     np.random.seed(1107)

#     x = np.random.randn(32, 255, 768) * 3
#     x = x.astype("float32")

#     x_paddle = paddle.to_tensor(x)
#     x_torch = torch.from_numpy(x).cuda()

#     n_heads = 12
#     d_model = 768
#     dropout = 0

#     # -------- torch --------
#     self_attn_torch = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).cuda() # <--- 注意这里是 batch first
#     # -------- paddle --------
#     self_attn_paddle = paddle.nn.MultiHeadAttention(d_model, n_heads, dropout=dropout)

#     # torch 的参数给 paddle 模型
#     self_attn_paddle = MultiheadAttention_torch2paddle(self_attn_paddle, self_attn_torch)


#     # torch
#     tgt2_torch = self_attn_torch(x_torch, x_torch, x_torch)[0]

#     # paddle
#     tgt2_paddle = self_attn_paddle(x_paddle, x_paddle, x_paddle)


#     print(  
#         tgt2_torch.mean().item() - tgt2_paddle.mean().item(),
#         tgt2_torch.std().item()  - tgt2_paddle.std().item(),
#     )



class FeatureExtractor(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv0 = paddle.nn.Conv1D(1, 512, 10, 5, bias_attr=False)
        self.norm0 = paddle.nn.GroupNorm(512, 512)
        self.conv1 = paddle.nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv2 = paddle.nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv3 = paddle.nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv4 = paddle.nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv5 = paddle.nn.Conv1D(512, 512, 2, 2, bias_attr=False)
        self.conv6 = paddle.nn.Conv1D(512, 512, 2, 2, bias_attr=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = paddle.nn.functional.gelu(self.norm0(self.conv0(x)))
        x = paddle.nn.functional.gelu(self.conv1(x))
        x = paddle.nn.functional.gelu(self.conv2(x))
        x = paddle.nn.functional.gelu(self.conv3(x))
        x = paddle.nn.functional.gelu(self.conv4(x))
        x = paddle.nn.functional.gelu(self.conv5(x))
        x = paddle.nn.functional.gelu(self.conv6(x))
        return x



class FeatureExtractor_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = torch.nn.GroupNorm(512, 512)
        self.conv1 = torch.nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = torch.nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = torch.nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = torch.nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = torch.nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = torch.nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.norm0(self.conv0(x)))
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = torch.nn.functional.gelu(self.conv3(x))
        x = torch.nn.functional.gelu(self.conv4(x))
        x = torch.nn.functional.gelu(self.conv5(x))
        x = torch.nn.functional.gelu(self.conv6(x))
        return x



# if __name__ == "__main__":
#     # 目标, 将 torch 参数传递给 paddle 模型
#     torch_fe = FeatureExtractor_torch().cuda()
#     paddle_fe = FeatureExtractor()

    
#     np.random.seed(1107)
#     inputs = np.random.rand(1, 1, 574480).astype("float32")

#     tc_inp = torch.from_numpy(inputs).cuda()
#     pd_inp = paddle.to_tensor(inputs)


#     paddle_fe_state_dict = paddle_fe.state_dict()
#     # 目测参数都一样, 直接转换就可以
#     if set(torch_fe.state_dict().keys()) == set(paddle_fe.state_dict().keys()):

#         for torch_key, torch_value in torch_fe.state_dict().items():

#             paddle_fe_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() )
    
#     else:
#         print("WTF!?")

#     paddle_fe.load_dict(paddle_fe_state_dict)

#     # 运行模型
#     y_tc = torch_fe(tc_inp)
#     y_pd = paddle_fe(pd_inp)

#     print(
#         abs((y_tc.cpu().detach().numpy()
#          -
#         y_pd.numpy())).max().item(),
#     )



class FeatureProjection(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.norm = paddle.nn.LayerNorm(512)
        self.projection = paddle.nn.Linear(512, 768)
        self.dropout = paddle.nn.Dropout(0.1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class FeatureProjection_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(512)
        self.projection = torch.nn.Linear(512, 768)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x



# if __name__ == "__main__":
#     # 目标, 将 torch 参数传递给 paddle 模型

#     torch_fp = FeatureProjection_torch().cuda()
#     paddle_fp = FeatureProjection()

#     # 开 eval , 有 dropout 
#     torch_fp.eval()
#     paddle_fp.eval()

    
#     np.random.seed(1107)
#     inputs = np.random.rand(1, 1795, 512).astype("float32")

#     tc_inp = torch.from_numpy(inputs).cuda()
#     pd_inp = paddle.to_tensor(inputs)


#     paddle_fp_state_dict = paddle_fp.state_dict()
#     # 目测参数都一样, 直接转换就可以
#     if set(torch_fp.state_dict().keys()) == set(paddle_fp.state_dict().keys()):

#         for torch_key, torch_value in torch_fp.state_dict().items():

#             if "projection.weight" == torch_key:
#                 paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() ).T
#             else:
#                 assert paddle_fp_state_dict[torch_key].shape == list(torch_value.shape) 
#                 paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() )
    
#     else:
#         print("WTF!?")


#     paddle_fp.load_dict(paddle_fp_state_dict)

#     # 运行模型
#     y_tc = torch_fp(tc_inp)
#     y_pd = paddle_fp(pd_inp)

#     print(
#         abs((y_tc.cpu().detach().numpy()
#          -
#         y_pd.numpy())).max().item(),
#     )



class PositionalConvEmbedding(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv1D(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = paddle.nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x.transpose([0, 2, 1]))
        x = paddle.nn.functional.gelu(x[:, :, :-1])
        return x.transpose([0, 2, 1])


class PositionalConvEmbedding_torch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = torch.nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = torch.nn.functional.gelu(x[:, :, :-1])
        return x.transpose(1, 2)



# if __name__ == "__main__":
#     # 目标, 将 torch 参数传递给 paddle 模型

#     torch_fp = PositionalConvEmbedding_torch().cuda()
#     paddle_fp = PositionalConvEmbedding()

#     # 开 eval , 有 dropout 
#     torch_fp.eval()
#     paddle_fp.eval()

    
#     np.random.seed(1107)
#     inputs = np.random.rand(16, 1795, 768).astype("float32")

#     tc_inp = torch.from_numpy(inputs).cuda()
#     pd_inp = paddle.to_tensor(inputs)

#     paddle_fp_state_dict = paddle_fp.state_dict()
#     # 目测参数都一样, 直接转换就可以
#     if set(torch_fp.state_dict().keys()) == set(paddle_fp.state_dict().keys()):

#         for torch_key, torch_value in torch_fp.state_dict().items():

#             # assert paddle_fp_state_dict[torch_key].shape == list(torch_value.shape) 
#             assert paddle_fp_state_dict[torch_key].size == torch_value.numel()

#             origin_shape = paddle_fp_state_dict[torch_key].shape

#             paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() ).reshape(origin_shape)
    
#     else:
#         print("WTF!?")


#     paddle_fp.load_dict(paddle_fp_state_dict)

#     # 运行模型
#     y_tc = torch_fp(tc_inp)
#     y_pd = paddle_fp(pd_inp)

#     print(
#         "PositionalConvEmbedding",
#         abs((y_tc.cpu().detach().numpy()
#          -
#         y_pd.numpy())).max().item(),
#     )



class TransformerEncoder(paddle.nn.Layer):
    def __init__(
            self, encoder_layer: paddle.nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = paddle.nn.LayerList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
            self,
            src: paddle.Tensor,
            mask: paddle.Tensor = None,
            src_key_padding_mask: paddle.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> paddle.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, 
                # src_key_padding_mask=src_key_padding_mask # <-------- paddle 没有这个参数
            )
        return output


class TransformerEncoder_torch(torch.nn.Module):
    def __init__(
            self, encoder_layer: torch.nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder_torch, self).__init__()
        self.layers = torch.nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
            self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


# if __name__ == "__main__":

#     torch_encoder = TransformerEncoder_torch(
#         torch.nn.TransformerEncoderLayer(
#             768, 12, 3072, activation="gelu", batch_first=True
#         ),
#         12,
#     ).cuda()

#     paddle_encoder = TransformerEncoder(
#         paddle.nn.TransformerEncoderLayer(
#             768, 12, 3072, activation="gelu"
#         ),
#         12,
#     )

#     torch_encoder.eval()
#     paddle_encoder.eval()

#     np.random.seed(1107)
#     inputs = np.random.rand(2, 1795, 768).astype("float32") # <--- 

#     tc_inp = torch.from_numpy(inputs).cuda()
#     pd_inp = paddle.to_tensor(inputs)

#     paddle_fp_state_dict = paddle_encoder.state_dict()

#     # torch 参数
#     param_torch = set(torch_encoder.state_dict().keys())
#     param_paddle = set(paddle_encoder.state_dict().keys())

#     # 所有 torch 有, 但 paddle 没有的参数
#     # 这个主要是 multihead attention 那块儿参数的问题
#     param_torch - param_paddle

#     pattern = r"layers\.\d+\.linear\d+\.weight"

#     # 先解决共有参数的问题
#     for torch_key in param_torch & param_paddle:

#         torch_value = torch_encoder.state_dict()[torch_key]
#         assert paddle_fp_state_dict[torch_key].size == torch_value.numel()

#         match = re.match(pattern, torch_key)
#         if match:
#             # 匹配到了, 需要转置
#             # print(torch_key)
#             paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy().T )
#         else:
#             # 没有匹配到了
#             assert paddle_fp_state_dict[torch_key].shape == list(torch_value.shape) 
#             paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() )
            
#     assert len(param_paddle - param_torch) == len(param_torch - param_paddle) * 3

#     # 参数加载一遍
#     paddle_encoder.load_dict(paddle_fp_state_dict)

#     # 接下来就是 attention 部分的参数转换
#     for idx in range( len(paddle_encoder.layers) ):

#         pd_model = paddle_encoder.layers[idx].self_attn
#         tc_model = torch_encoder.layers[idx].self_attn

#         prefix = f"layers.{idx}.self_attn."

#         print("BEFORE ", pd_model.q_proj.weight.data.mean().item())
#         paddle_encoder.layers[idx].self_attn = MultiheadAttention_torch2paddle(pd_model, tc_model, prefix="")
#         print("AFTER ", pd_model.q_proj.weight.data.mean().item())

#     # param_torch = set(torch_encoder.state_dict().keys())
#     # param_paddle = set(paddle_encoder.state_dict().keys())

#     # 运行模型
#     y_tc = torch_encoder(tc_inp, output_layer=None)
#     y_pd = paddle_encoder(pd_inp, output_layer=None)

#     print(
#         "TransformerEncoder",
#         abs((y_tc.cpu().detach().numpy()
#          -
#         y_pd.numpy())).max().item(),
#     )


def TransformerEncoder_torch2paddle(torch_encoder, paddle_encoder):

    paddle_fp_state_dict = paddle_encoder.state_dict()

    # torch 参数
    param_torch = set(torch_encoder.state_dict().keys())
    param_paddle = set(paddle_encoder.state_dict().keys())

    # 所有 torch 有, 但 paddle 没有的参数
    # 这个主要是 multihead attention 那块儿参数的问题
    param_torch - param_paddle

    pattern = r"layers\.\d+\.linear\d+\.weight"

    # 先解决共有参数的问题
    for torch_key in param_torch & param_paddle:

        torch_value = torch_encoder.state_dict()[torch_key]
        assert paddle_fp_state_dict[torch_key].size == torch_value.numel()

        match = re.match(pattern, torch_key)
        if match:
            # 匹配到了, 需要转置
            # print(torch_key)
            paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy().T )
        else:
            # 没有匹配到了
            assert paddle_fp_state_dict[torch_key].shape == list(torch_value.shape) 
            paddle_fp_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() )
            
    assert len(param_paddle - param_torch) == len(param_torch - param_paddle) * 3

    # 参数加载一遍
    paddle_encoder.load_dict(paddle_fp_state_dict)

    # 接下来就是 attention 部分的参数转换
    for idx in range( len(paddle_encoder.layers) ):

        pd_model = paddle_encoder.layers[idx].self_attn
        tc_model = torch_encoder.layers[idx].self_attn

        prefix = f"layers.{idx}.self_attn."

        print("BEFORE ", pd_model.q_proj.weight.data.mean().item())
        paddle_encoder.layers[idx].self_attn = MultiheadAttention_torch2paddle(pd_model, tc_model, prefix="")
        print("AFTER ", pd_model.q_proj.weight.data.mean().item())

    # param_torch = set(torch_encoder.state_dict().keys())
    # param_paddle = set(paddle_encoder.state_dict().keys())

    return paddle_encoder



# # 测试 TransformerEncoder_torch2paddle
# if __name__ == "__main__":

#     np.random.seed(1107)
#     inputs = np.random.rand(2, 1795, 768).astype("float32") # <--- 

#     tc_inp = torch.from_numpy(inputs).to("cuda:1")
#     pd_inp = paddle.to_tensor(inputs)

#     torch_encoder = TransformerEncoder_torch(
#         torch.nn.TransformerEncoderLayer(
#             768, 12, 3072, activation="gelu", batch_first=True
#         ),
#         12,
#     ).to("cuda:1")

#     paddle_encoder = TransformerEncoder(
#         paddle.nn.TransformerEncoderLayer(
#             768, 12, 3072, activation="gelu"
#         ),
#         12,
#     )

#     torch_encoder.eval()
#     paddle_encoder.eval()

#     paddle_encoder = TransformerEncoder_torch2paddle(torch_encoder, paddle_encoder)


#     # 运行模型
#     y_tc = torch_encoder(tc_inp, output_layer=None)
#     y_pd = paddle_encoder(pd_inp, output_layer=None)

#     print(
#         "TransformerEncoder",
#         abs((y_tc.cpu().detach().numpy()
#          -
#         y_pd.numpy())).max().item(),
#     )



def _compute_mask_torch(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask

# 推理不用
def _compute_mask(shape: Tuple[int, int], mask_prob: float, mask_length:
    int, min_masks: int=0) ->paddle.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')

    if mask_length > sequence_length:
        raise ValueError(
            f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`'
            )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length +
        random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length
    
    # SpecAugment mask to fill
    mask = paddle.zeros(shape=(batch_size, sequence_length), dtype='bool')

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = paddle.ones(shape=(batch_size, sequence_length - (
        mask_length - 1)))

    # get random indices to mask
    mask_indices = paddle.multinomial(x=uniform_dist, num_samples=
        num_masked_spans)
    
    # expand masked indices to masked spans
    mask_indices = mask_indices.unsqueeze(axis=-1).expand(shape=(batch_size,
        num_masked_spans, mask_length)).reshape([batch_size, 
        num_masked_spans * mask_length])
    offsets = paddle.arange(end=mask_length)[None, None, :].expand((
        batch_size, num_masked_spans, mask_length)).reshape([batch_size, 
        num_masked_spans * mask_length])
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.astype(int).put_along_axis(axis=1, indices=mask_idxs, values=1, broadcast=False).astype(bool)

    return mask


# if __name__ == "__main__":

#     torch_param = ((1, 11), 0.8, 10, 'cuda:0', 2)
#     paddle_param = ((1, 11), 0.8, 10, 2)

#     torch_out = _compute_mask_torch(*torch_param)
#     paddle_out = _compute_mask(*paddle_param)

#     # --------- multinomial 这里是随机的 ---------
#     # print(
#     #     (torch_out.cpu().numpy() == paddle_out.numpy()).all()
#     # )




class Hubert(paddle.nn.Layer):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = paddle.nn.LayerNorm(768)
        self.dropout = paddle.nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            paddle.nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", 
                # batch_first=True # <-------- paddle 默认 batch first
            ),
            12,
        )
        self.proj = paddle.nn.Linear(768, 256)
        self.masked_spec_embed = paddle.create_parameter(
            shape=[768], 
            dtype='float32', 
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[768], dtype='float32').uniform_()))

        self.label_embedding = paddle.nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
            self, x: paddle.Tensor, layer: Optional[int] = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose([0, 2, 1]))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    # def logits(self, x: paddle.Tensor) -> paddle.Tensor:
    #     logits = paddle.nn.functional.cosine_similarity(
    #         x.unsqueeze(2),
    #         self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
    #         axis=-1,
    #     )
    #     return logits / 0.1

    # def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    #     x, mask = self.encode(x)
    #     x = self.proj(x)
    #     logits = self.logits(x)
    #     return logits, mask


class Hubert_torch(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor_torch()
        self.feature_projection = FeatureProjection_torch()
        self.positional_embedding = PositionalConvEmbedding_torch()
        self.norm = torch.nn.LayerNorm(768)
        self.dropout = torch.nn.Dropout(0.1)
        self.encoder = TransformerEncoder_torch(
            torch.nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        self.proj = torch.nn.Linear(768, 256)

        self.masked_spec_embed = torch.nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = torch.nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask_torch((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
            self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    # def logits(self, x: torch.Tensor) -> torch.Tensor:
    #     logits = torch.cosine_similarity(
    #         x.unsqueeze(2),
    #         self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
    #         dim=-1,
    #     )
    #     return logits / 0.1

    # def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     x, mask = self.encode(x)
    #     x = self.proj(x)
    #     logits = self.logits(x)
    #     return logits, mask



class HubertSoft_torch(Hubert_torch):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)


class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    @paddle.no_grad()
    def units(self, wav: paddle.Tensor) -> paddle.Tensor:
        wav = paddle.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2), data_format="NCL")
        x, _ = self.encode(wav)
        return self.proj(x)


# # -------------------------------- 测试 HubertSoft --------------------------------
# if __name__ == "__main__":

#     x = np.random.rand(1, 1, 574400).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     m_tc = HubertSoft_torch().cuda()
#     m_pd = HubertSoft()

#     m_tc.eval() # for dropout
#     m_pd.eval()


#     # ------------ 参数集合 ------------
#     m_tc_params = set(m_tc.state_dict().keys())
#     m_pd_params = set(m_pd.state_dict().keys())

#     pattern = r"encoder.layers\.\d+\.linear\d+\.weight"

#     m_pd_state_dict = m_pd.state_dict()

#     # 先解决共有参数的问题
#     for torch_key in m_tc_params & m_pd_params:

#         # print(torch_key)

#         torch_value = m_tc.state_dict()[torch_key]
#         assert m_pd.state_dict()[torch_key].size == torch_value.numel()

#         match = re.match(pattern, torch_key)
#         if match or torch_key in ["feature_projection.projection.weight", "proj.weight"]:
#             # 匹配到了, 需要转置
#             # print(torch_key)
#             m_pd_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy().T )
#         else:
            
#             if torch_key.endswith('.weight_g'):
#                 m_pd_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() ).reshape(
#                     m_pd_state_dict[torch_key].shape
#                 )

#             else:
#                 # 没有匹配到了
#                 assert m_pd_state_dict[torch_key].shape == list(torch_value.shape) 
#                 m_pd_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() )

#     # 参数加载一遍
#     m_pd.load_dict(m_pd_state_dict)

#     assert len(m_pd_params - m_tc_params) == len(m_tc_params - m_pd_params) * 3

#     m_pd.encoder = TransformerEncoder_torch2paddle(m_tc.encoder, m_pd.encoder)

#     y_tc = m_tc.units(x_tc)
#     y_pd = m_pd.units(x_pd)

#     print(
#         abs(
#             y_tc.detach().cpu().numpy()
#             -
#             y_pd.numpy()
#         ).max().item()
#     )



def hubert_soft_torch2paddle(m_tc, m_pd):

    m_tc.eval() # for dropout
    m_pd.eval()

    # ------------ 参数集合 ------------
    m_tc_params = set(m_tc.state_dict().keys())
    m_pd_params = set(m_pd.state_dict().keys())

    pattern = r"encoder.layers\.\d+\.linear\d+\.weight"

    m_pd_state_dict = m_pd.state_dict()

    # 先解决共有参数的问题
    for torch_key in m_tc_params & m_pd_params:

        # print(torch_key)

        torch_value = m_tc.state_dict()[torch_key]
        assert m_pd.state_dict()[torch_key].size == torch_value.numel()

        match = re.match(pattern, torch_key)
        if match or torch_key in ["feature_projection.projection.weight", "proj.weight"]:
            # 匹配到了, 需要转置
            # print(torch_key)
            m_pd_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy().T )
        else:
            
            if torch_key.endswith('.weight_g'):
                m_pd_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() ).reshape(
                    m_pd_state_dict[torch_key].shape
                )

            else:
                # 没有匹配到了
                assert m_pd_state_dict[torch_key].shape == list(torch_value.shape) 
                m_pd_state_dict[torch_key] = paddle.to_tensor( torch_value.detach().cpu().numpy() )

    # 参数加载一遍
    m_pd.load_dict(m_pd_state_dict)

    assert len(m_pd_params - m_tc_params) == len(m_tc_params - m_pd_params) * 3

    m_pd.encoder = TransformerEncoder_torch2paddle(m_tc.encoder, m_pd.encoder)

    return m_pd


# # -------------------------------- 测试 hubert_soft_torch2paddle --------------------------------
# if __name__ == "__main__":

#     x = np.random.rand(1, 1, 574400).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     m_tc = HubertSoft_torch().cuda()
#     m_pd = HubertSoft()

#     m_tc.eval() # for dropout
#     m_pd.eval()

#     m_pd = hubert_soft_torch2paddle(m_tc, m_pd)

#     y_tc = m_tc.units(x_tc)
#     y_pd = m_pd.units(x_pd)

#     print(
#         "hubert_soft_torch2paddle",
#         abs(
#             y_tc.detach().cpu().numpy()
#             -
#             y_pd.numpy()
#         ).max().item()
#     )




def hubert_soft_torch(
        path: str,
) -> HubertSoft_torch:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft_torch()
    checkpoint = torch.load(path)

    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.load_state_dict(checkpoint)
    hubert.eval()
    return hubert



def hubert_soft(
        path: str,
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft()
    checkpoint = paddle.load(path)
    # consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.load_dict(checkpoint)
    hubert.eval()
    return hubert


# if __name__ == "__main__":

#     model_path = "~/Desktop/whisper-vits-svc/hubert_pretrain/hubert-soft-0d54a1f4.pt"
#     model_path = os.path.expanduser(model_path)
#     tc_model = hubert_soft_torch(model_path)

#     pd_model = HubertSoft()
#     pd_model = hubert_soft_torch2paddle(tc_model, pd_model)

#     paddle.save(
#         pd_model.state_dict(),
#         "hubert_pretrain/hubert-soft.pdparam"
#     )

#     model_path = "hubert_pretrain/hubert-soft.pdparam"
#     hubert_soft(model_path)