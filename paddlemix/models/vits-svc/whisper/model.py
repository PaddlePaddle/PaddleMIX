from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import paddle
import torch
# import torch.nn.functional as F
# from torch import torch.Tensor
# from torch import nn

# from .decoding import detect_language as detect_language_function, decode as decode_function


def init_weights(model):
    for param in model.parameters():
        # print(param.shape)
        torch.nn.init.uniform_(param)

def LayerNorm_torch2paddle(model_torch, model_paddle):

    model_paddle.weight.set_value( 
        model_torch.weight.data.cpu().numpy() 
        )
    model_paddle.bias.set_value( 
        model_torch.bias.data.cpu().numpy() 
        )

# if __name__ == "__main__":

#     model_tc = torch.nn.LayerNorm(256).cuda()
#     init_weights(model_tc)
#     model_pd = paddle.nn.LayerNorm(256)

#     LayerNorm_torch2paddle(model_tc, model_pd)

#     x = np.random.randn(1, 1500, 256).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     y_tc = model_tc(x_tc)
#     y_pd = model_pd(x_pd)

#     y_tc = y_tc.detach().cpu().numpy()
#     y_pd = y_pd.detach().cpu().numpy()

#     print(
#         abs(y_tc - y_pd).max(),
#     )




@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


# class LayerNorm(torch.nn.LayerNorm):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # return super().forward(x.float()).type(x.dtype) sovits5.0
#         return super().forward(x).type(x.dtype)


# class Linear(torch.nn.Linear):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return F.linear(
#             x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
#         )


# class Conv1d(torch.nn.Conv1d):
#     def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
#         return super()._conv_forward(
#             x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
#         )


def sinusoids_torch(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = paddle.exp(-log_timescale_increment * paddle.arange(channels // 2))
    scaled_time = paddle.arange(length)[:, np.newaxis].astype("float32") * inv_timescales[np.newaxis, :]
    return paddle.concat([paddle.sin(scaled_time), paddle.cos(scaled_time)], axis=1)


# if __name__ == "__main__":

#     y_tc = sinusoids_torch(1500, 1280)
#     y_pd = sinusoids(1500, 1280)

#     y_delta = y_pd.cpu().numpy() - y_tc.cpu().numpy()

#     print(
#         abs(y_delta).max()
#     )

    



class MultiHeadAttention_torch(torch.nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = torch.nn.Linear(n_state, n_state)
        self.key = torch.nn.Linear(n_state, n_state, bias=False)
        self.value = torch.nn.Linear(n_state, n_state)
        self.out = torch.nn.Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv torch.Tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttention(paddle.nn.Layer):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = paddle.nn.Linear(n_state, n_state)
        self.key = paddle.nn.Linear(n_state, n_state, bias_attr=False)
        self.value = paddle.nn.Linear(n_state, n_state)
        self.out = paddle.nn.Linear(n_state, n_state)

    def forward(
        self,
        x: paddle.Tensor,
        xa: Optional[paddle.Tensor] = None,
        mask: Optional[paddle.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv torch.Tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: paddle.Tensor, 
                            k: paddle.Tensor, 
                            v: paddle.Tensor, 
                            mask: Optional[paddle.Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape([*q.shape[:2], self.n_head, -1]).transpose([0, 2, 1, 3]) * scale
        k = k.reshape([*k.shape[:2], self.n_head, -1]).transpose([0, 2, 3, 1]) * scale
        v = v.reshape([*v.shape[:2], self.n_head, -1]).transpose([0, 2, 1, 3])

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.astype("float32")

        w = paddle.nn.functional.softmax(qk, axis=-1).astype(q.dtype)
        return (w @ v).transpose([0, 2, 1, 3]).flatten(start_axis=2), qk.detach()


def MultiHeadAttention_torch2paddle(model_torch, model_paddle):

    model_paddle.query.weight.set_value( model_torch.query.weight.data.T.cpu().numpy() )
    model_paddle.query.bias.set_value( model_torch.query.bias.data.cpu().numpy() )

    model_paddle.key.weight.set_value( model_torch.key.weight.data.T.cpu().numpy() )
    # model_paddle.key.bias.set_value( model_torch.query.weight.data )

    model_paddle.value.weight.set_value( model_torch.value.weight.data.T.cpu().numpy() )
    model_paddle.value.bias.set_value( model_torch.value.bias.data.cpu().numpy() )
    model_paddle.out.weight.set_value( model_torch.out.weight.data.T.cpu().numpy() )
    model_paddle.out.bias.set_value( model_torch.out.bias.data.cpu().numpy() )

    # attn_ln



# if __name__ == "__main__":

#     model_tc = MultiHeadAttention_torch(1280, 20).cuda()
#     model_pd = MultiHeadAttention(1280, 20)

#     x = np.random.randn(1, 1500, 1280).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     MultiHeadAttention_torch2paddle(model_tc, model_pd)

#     y_tc1, y_tc2 = model_tc(x_tc)
#     y_pd1, y_pd2 = model_pd(x_pd)

#     y_tc1, y_tc2 = y_tc1.detach().cpu().numpy(), y_tc2.detach().cpu().numpy()
#     y_pd1, y_pd2 = y_pd1.detach().cpu().numpy(), y_pd2.detach().cpu().numpy()

#     print(
#         abs(y_tc1 - y_pd1).max(),

#         abs(y_tc2 - y_pd2).max(),
#     )









class ResidualAttentionBlock_torch(torch.nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention_torch(n_state, n_head)
        self.attn_ln = torch.nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention_torch(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = torch.nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_mlp), 
            torch.nn.GELU(), 
            torch.nn.Linear(n_mlp, n_state)
            )
        self.mlp_ln = torch.nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x



class ResidualAttentionBlock(paddle.nn.Layer):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = paddle.nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = paddle.nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = paddle.nn.Sequential(
                            paddle.nn.Linear(n_state, n_mlp), 
                            paddle.nn.GELU(), 
                            paddle.nn.Linear(n_mlp, n_state)
                            )
        self.mlp_ln = paddle.nn.LayerNorm(n_state)

    def forward(
        self,
        x: paddle.Tensor,
        xa: Optional[paddle.Tensor] = None,
        mask: Optional[paddle.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x



if __name__ == "__main__":

    x = np.random.rand(1, 1499, 1280).astype("float32")
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)

    model_tc = ResidualAttentionBlock_torch(1280, 20, False).cuda()
    # init_weights(model_tc)

    model_pd = ResidualAttentionBlock(1280, 20, False)

    model_tc_state_dict = model_tc.state_dict()
    model_pd_state_dict = model_pd.state_dict()

    print(
        set( model_tc.state_dict().keys() ) == set( model_pd.state_dict().keys() )
    )

    for torch_key, torch_value in model_pd.state_dict().items():
        if list(torch_value.shape) == model_pd_state_dict[torch_key].shape:
            model_pd_state_dict[torch_key] = paddle.to_tensor(
                torch_value.detach().cpu().numpy()
            )
        else:
            print(torch_key)
    
    model_pd.set_state_dict( model_pd_state_dict )

    MultiHeadAttention_torch2paddle(model_tc.attn, model_pd.attn)
    # MultiHeadAttention_torch2paddle(model_tc.cross_attn, model_pd.cross_attn)

    model_pd.mlp[0].weight.set_value( 
        paddle.to_tensor(
            model_tc.mlp[0].weight.data.cpu().numpy().T
        )
    )
    model_pd.mlp[0].bias.set_value( 
        paddle.to_tensor(
            model_tc.mlp[0].bias.data.cpu().numpy()
        )
    )

    model_pd.mlp[2].weight.set_value( 
        paddle.to_tensor(
            model_tc.mlp[2].weight.data.cpu().numpy().T
        )
    )
    model_pd.mlp[2].bias.set_value( 
        paddle.to_tensor(
            model_tc.mlp[2].bias.data.cpu().numpy()
        )
    )

    # ----------- 一些 LayerNorm -----------
    model_pd.mlp_ln.weight.set_value(
        paddle.to_tensor(
            model_tc.mlp_ln.weight.data.cpu().numpy()
        )
    )
    model_pd.mlp_ln.bias.set_value(
        paddle.to_tensor(
            model_tc.mlp_ln.bias.data.cpu().numpy()
        )
    )

    model_pd.attn_ln.weight.set_value(
        paddle.to_tensor(
            model_tc.attn_ln.weight.data.cpu().numpy()
        )
    )
    model_pd.attn_ln.bias.set_value(
        paddle.to_tensor(
            model_tc.attn_ln.bias.data.cpu().numpy()
        )
    )

    y_tc = model_tc(x_tc).detach().cpu().numpy()
    y_pd = model_pd(x_pd).detach().cpu().numpy()

    print(
        abs(y_tc - y_pd).max()
    )



def ResidualAttentionBlock_torch2paddle(model_tc, model_pd):

    model_tc_state_dict = model_tc.state_dict()
    model_pd_state_dict = model_pd.state_dict()

    # print(
    #     set( model_tc.state_dict().keys() ) == set( model_pd.state_dict().keys() )
    # )

    for torch_key, torch_value in model_pd.state_dict().items():
        if list(torch_value.shape) == model_pd_state_dict[torch_key].shape:
            model_pd_state_dict[torch_key] = paddle.to_tensor(
                torch_value.detach().cpu().numpy()
            )
        else:
            print(torch_key)
    
    model_pd.set_state_dict( model_pd_state_dict )

    MultiHeadAttention_torch2paddle(model_tc.attn, model_pd.attn)
    # MultiHeadAttention_torch2paddle(model_tc.cross_attn, model_pd.cross_attn)

    model_pd.mlp[0].weight.set_value( 
        paddle.to_tensor(
            model_tc.mlp[0].weight.data.cpu().numpy().T
        )
    )
    model_pd.mlp[0].bias.set_value( 
        paddle.to_tensor(
            model_tc.mlp[0].bias.data.cpu().numpy()
        )
    )

    model_pd.mlp[2].weight.set_value( 
        paddle.to_tensor(
            model_tc.mlp[2].weight.data.cpu().numpy().T
        )
    )
    model_pd.mlp[2].bias.set_value( 
        paddle.to_tensor(
            model_tc.mlp[2].bias.data.cpu().numpy()
        )
    )

    LayerNorm_torch2paddle( model_tc.attn_ln, model_pd.attn_ln )
    LayerNorm_torch2paddle( model_tc.mlp_ln, model_pd.mlp_ln )



# if __name__ == "__main__":

#     x = np.random.rand(1, 1499, 1280).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     model_tc = ResidualAttentionBlock_torch(1280, 20, False).cuda()
#     # init_weights(model_tc)

#     model_pd = ResidualAttentionBlock(1280, 20, False)

#     ResidualAttentionBlock_torch2paddle(model_tc, model_pd)
    
#     y_tc = model_tc(x_tc).detach().cpu().numpy()
#     y_pd = model_pd(x_pd).detach().cpu().numpy()

#     print(
#         abs(y_tc - y_pd).max()
#     )




class AudioEncoder_torch(torch.nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids_torch(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock_torch] = torch.nn.ModuleList(
            [ResidualAttentionBlock_torch(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = torch.nn.LayerNorm(n_state)

        

    def forward(self, x: torch.Tensor):
        """
        x : torch.torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        len_x = x.shape[1]
        len_e = self.positional_embedding.shape[0]
        assert len_x <= len_e, "incorrect audio shape"
        pos_e = self.positional_embedding[:len_x, :]
        x = (x + pos_e).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class AudioEncoder(paddle.nn.Layer):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.conv1 = paddle.nn.Conv1D(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = paddle.nn.Conv1D(n_state, n_state, kernel_size=3, stride=2,
            padding=1)
        self.register_buffer(name='positional_embedding', tensor=
            sinusoids(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = paddle.nn.LayerList(
            sublayers=[ResidualAttentionBlock(n_state, n_head) for _ in
            range(n_layer)])
        self.ln_post = paddle.nn.LayerNorm(n_state)


    def forward(self, x: torch.Tensor):
        """
        x : torch.torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = paddle.nn.functional.gelu(self.conv1(x))
        x = paddle.nn.functional.gelu(self.conv2(x))
        x = x.transpose([0, 2, 1])

        len_x = x.shape[1]
        len_e = self.positional_embedding.shape[0]
        assert len_x <= len_e, "incorrect audio shape"
        pos_e = self.positional_embedding[:len_x, :]
        x = (x + pos_e).astype(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x



def AudioEncoder_torch2paddle(model_torch, model_paddle):

    model_paddle.conv1.weight.set_value(
        paddle.to_tensor(
            model_torch.conv1.weight.data.detach().cpu().numpy()
        )
    )
    model_paddle.conv1.bias.set_value(
        paddle.to_tensor(
            model_torch.conv1.bias.data.detach().cpu().numpy()
        )
    )

    model_paddle.conv2.weight.set_value(
        paddle.to_tensor(
            model_torch.conv2.weight.data.detach().cpu().numpy()
        )
    )
    model_paddle.conv2.bias.set_value(
        paddle.to_tensor(
            model_torch.conv2.bias.data.detach().cpu().numpy()
        )
    )

    model_paddle.ln_post.weight.set_value(
        paddle.to_tensor(
            model_torch.ln_post.weight.data.detach().cpu().numpy()
        )
    )
    model_paddle.ln_post.bias.set_value(
        paddle.to_tensor(
            model_torch.ln_post.bias.data.detach().cpu().numpy()
        )
    )

    for i in range(len(model_paddle.blocks)):
        ResidualAttentionBlock_torch2paddle(
            model_torch.blocks[i],
            model_paddle.blocks[i]
        ) 


# if __name__ == "__main__":

#     model_tc = AudioEncoder_torch(80, 1500, 1280, 20, 4).cuda()
#     model_pd = AudioEncoder(80, 1500, 1280, 20, 4)

#     x = np.random.rand(1, 80, 3000).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     AudioEncoder_torch2paddle(model_tc, model_pd)

#     y_tc = model_tc( x_tc ).detach().cpu().numpy()
#     y_pd = model_pd( x_pd ).detach().cpu().numpy()

#     print(
#         abs(y_tc - y_pd).max()
#     )










# class TextDecoder(torch.nn.Module):
#     def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
#         super().__init__()

#         self.token_embedding = nn.Embedding(n_vocab, n_state)
#         self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

#         self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
#             [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
#         )
#         self.ln = LayerNorm(n_state)

#         mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
#         self.register_buffer("mask", mask, persistent=False)

#     def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
#         """
#         x : torch.Longtorch.Tensor, shape = (batch_size, <= n_ctx)
#             the text tokens
#         xa : torch.torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
#             the encoded audio features to be attended on
#         """
#         offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
#         x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
#         x = x.to(xa.dtype)

#         for block in self.blocks:
#             x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

#         x = self.ln(x)
#         logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

#         return logits


class Whisper(paddle.nn.Layer):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # self.decoder = TextDecoder(
        #     self.dims.n_vocab,
        #     self.dims.n_text_ctx,
        #     self.dims.n_text_state,
        #     self.dims.n_text_head,
        #     self.dims.n_text_layer,
        # )

    # def embed_audio(self, mel: torch.torch.Tensor):
    #     return self.encoder(mel)

    # def logits(self, tokens: torch.torch.Tensor, audio_features: torch.torch.Tensor):
    #     return self.decoder(tokens, audio_features)

    # def forward(self, mel: torch.torch.Tensor, tokens: torch.torch.Tensor) -> Dict[str, torch.torch.Tensor]:
    #     return self.decoder(tokens, self.encoder(mel))

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    # @property
    # def is_multilingual(self):
    #     return self.dims.n_vocab == 51865

    # def install_kv_cache_hooks(self, cache: Optional[dict] = None):
    #     """
    #     The `MultiHeadAttention_torch` module optionally accepts `kv_cache` which stores the key and value
    #     torch.Tensors calculated for the previous positions. This method returns a dictionary that stores
    #     all caches, and the necessary hooks for the key and value projection modules that save the
    #     intermediate torch.Tensors to be reused during later calculations.

    #     Returns
    #     -------
    #     cache : Dict[nn.Module, torch.torch.Tensor]
    #         A dictionary object mapping the key/value projection modules to its cache
    #     hooks : List[RemovableHandle]
    #         List of PyTorch RemovableHandle objects to stop the hooks to be called
    #     """
    #     cache = {**cache} if cache is not None else {}
    #     hooks = []

    #     def save_to_cache(module, _, output):
    #         if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
    #             cache[module] = output  # save as-is, for the first token or cross attention
    #         else:
    #             cache[module] = torch.cat([cache[module], output], dim=1).detach()
    #         return cache[module]

    #     def install_hooks(layer: nn.Module):
    #         if isinstance(layer, MultiHeadAttention_torch):
    #             hooks.append(layer.key.register_forward_hook(save_to_cache))
    #             hooks.append(layer.value.register_forward_hook(save_to_cache))

    #     self.decoder.apply(install_hooks)
    #     return cache, hooks

    # detect_language = detect_language_function
    # decode = decode_function



class Whisper_torch(torch.nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder_torch(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # self.decoder = TextDecoder(
        #     self.dims.n_vocab,
        #     self.dims.n_text_ctx,
        #     self.dims.n_text_state,
        #     self.dims.n_text_head,
        #     self.dims.n_text_layer,
        # )

    # def embed_audio(self, mel: torch.torch.Tensor):
    #     return self.encoder(mel)

    # def logits(self, tokens: torch.torch.Tensor, audio_features: torch.torch.Tensor):
    #     return self.decoder(tokens, audio_features)

    # def forward(self, mel: torch.torch.Tensor, tokens: torch.torch.Tensor) -> Dict[str, torch.torch.Tensor]:
    #     return self.decoder(tokens, self.encoder(mel))

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    # @property
    # def is_multilingual(self):
    #     return self.dims.n_vocab == 51865

    # def install_kv_cache_hooks(self, cache: Optional[dict] = None):
    #     """
    #     The `MultiHeadAttention_torch` module optionally accepts `kv_cache` which stores the key and value
    #     torch.Tensors calculated for the previous positions. This method returns a dictionary that stores
    #     all caches, and the necessary hooks for the key and value projection modules that save the
    #     intermediate torch.Tensors to be reused during later calculations.

    #     Returns
    #     -------
    #     cache : Dict[nn.Module, torch.torch.Tensor]
    #         A dictionary object mapping the key/value projection modules to its cache
    #     hooks : List[RemovableHandle]
    #         List of PyTorch RemovableHandle objects to stop the hooks to be called
    #     """
    #     cache = {**cache} if cache is not None else {}
    #     hooks = []

    #     def save_to_cache(module, _, output):
    #         if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
    #             cache[module] = output  # save as-is, for the first token or cross attention
    #         else:
    #             cache[module] = torch.cat([cache[module], output], dim=1).detach()
    #         return cache[module]

    #     def install_hooks(layer: nn.Module):
    #         if isinstance(layer, MultiHeadAttention_torch):
    #             hooks.append(layer.key.register_forward_hook(save_to_cache))
    #             hooks.append(layer.value.register_forward_hook(save_to_cache))

    #     self.decoder.apply(install_hooks)
    #     return cache, hooks

    # detect_language = detect_language_function
    # decode = decode_function


checkpoint_dims = {
    'n_mels': 80,
    'n_vocab': 51865,
    'n_audio_ctx': 1500,
    'n_audio_state': 1280,
    'n_audio_head': 20,
    'n_audio_layer': 32,
    'n_text_ctx': 448,
    'n_text_state': 1280,
    'n_text_head': 20,
    'n_text_layer': 32,
}


if __name__ == "__main__":

    dims = ModelDimensions(**checkpoint_dims)

    model_tc = Whisper_torch(dims).cuda()
    model_pd = Whisper(dims)

    x = np.random.rand(1, 80, 1000).astype("float32")
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)

    AudioEncoder_torch2paddle(model_tc.encoder, model_pd.encoder)

    y_tc = model_tc.encoder( x_tc ).detach().cpu().numpy()
    y_pd = model_pd.encoder( x_pd ).detach().cpu().numpy()

    print(
        abs(y_tc - y_pd).max()
    )
