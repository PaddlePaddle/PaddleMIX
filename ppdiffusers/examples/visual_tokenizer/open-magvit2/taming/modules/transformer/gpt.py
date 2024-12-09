import sys
sys.path.append(
    './utils')
import paddle_aux
import paddle
"""
Refer to 
https://github.com/FoundationVision/LlamaGen
https://github.com/kakaobrain/rq-vae-transformer
"""
from typing import Optional
import math
from paddlenlp.utils.initializer import normal_,zeros_


def top_k_top_p_filtering(logits, top_k: int=0, top_p: float=1.0,
    filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])
        indices_to_remove = logits < paddle.topk(k=top_k, x=logits)[0][...,
            -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = paddle.sort(descending=True, x=logits
            ), paddle.argsort(descending=True, x=logits)
        cumulative_probs = paddle.cumsum(x=paddle.nn.functional.softmax(x=
            sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1
            ].clone()
        sorted_indices_to_remove[..., 0] = 0
        dtype = sorted_indices_to_remove.dtype
        indices_to_remove = sorted_indices_to_remove.astype('int32').put_along_axis(axis=1,
            indices=sorted_indices, values=sorted_indices_to_remove.astype('int32'),
            broadcast=False).astype(dtype)
        logits[indices_to_remove] = filter_value
    return logits


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - n % k


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


def drop_path(x, drop_prob: float=0.0, training: bool=False, scale_by_keep:
    bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (tuple(x.shape)[0],) + (1,) * (x.ndim - 1)
    """Class Method: *.bernoulli_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    random_tensor = paddle.empty(shape=shape, dtype=x.dtype).bernoulli_(
        keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.divide_(y=paddle.to_tensor(keep_prob))
    return x * random_tensor


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float=0.0, scale_by_keep: bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class LabelEmbedder(paddle.nn.Layer):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = paddle.nn.Embedding(num_embeddings=
            num_classes + use_cfg_embedding, embedding_dim=hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = paddle.rand(shape=tuple(labels.shape)[0]
                ) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = paddle.where(condition=drop_ids, x=self.num_classes, y=labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        labels = labels.squeeze(axis=-1)
        if train and use_dropout or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(axis=1)
        return embeddings


class MLP(paddle.nn.Layer):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(in_features=in_features, out_features=
            hidden_features, bias_attr=False)
        self.act = paddle.nn.GELU(approximate=True)
        self.fc2 = paddle.nn.Linear(in_features=hidden_features,
            out_features=out_features, bias_attr=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RMSNorm(paddle.nn.Layer):

    def __init__(self, dim: int, eps: float=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.ones(shape=dim))

    def _norm(self, x):
        return x * paddle.rsqrt(x=paddle.mean(x=x * x, axis=-1, keepdim=
            True) + self.eps)

    def forward(self, x):
        output = self._norm(x.astype(dtype='float32')).astype(dtype=x.dtype)
        return output * self.weight


class FeedForward(paddle.nn.Layer):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)
        self.w1 = paddle.nn.Linear(in_features=config.dim, out_features=
            hidden_dim, bias_attr=False)
        self.w3 = paddle.nn.Linear(in_features=config.dim, out_features=
            hidden_dim, bias_attr=False)
        self.w2 = paddle.nn.Linear(in_features=hidden_dim, out_features=
            config.dim, bias_attr=False)
        
        self.ffn_dropout = paddle.nn.Dropout(p=config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(paddle.nn.functional.silu(x=self.w1
            (x)) * self.w3(x)))


class Attention(paddle.nn.Layer):

    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = (config.n_kv_head if config.n_kv_head is not None else
            config.n_head)
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.wqkv = paddle.nn.Linear(in_features=config.dim, out_features=
            total_kv_dim, bias_attr=False)
        self.wo = paddle.nn.Linear(in_features=config.dim, out_features=
            config.dim, bias_attr=False)
        self.kv_cache = None
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = paddle.nn.Dropout(p=config.resid_dropout_p)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=
        None, dropout_p=0.0, is_causal=False, scale=None) ->paddle.Tensor:
        L, S = query.shape[-2], key.shape[-2]
        B,N = query.shape[0],attn_mask.shape[1]
        scale_factor = 1 / math.sqrt(query.shape[-1]
            ) if scale is None else scale
        attn_bias = paddle.zeros(shape=[B,N,L, S], dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = paddle.ones(shape=[L, S], dtype='bool').tril(diagonal=0
                )
            attn_bias.masked_fill_(mask=temp_mask.logical_not(), value=
                float('-inf'))
            attn_bias.to(query.dtype)
        if attn_mask is not None:
            if attn_mask.dtype == paddle.bool:
                attn_bias.masked_fill_(mask=attn_mask.logical_not(), value=
                    float('-inf'))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(perm=paddle_aux.
            transpose_aux_func(key.ndim, -2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = paddle.nn.functional.softmax(x=attn_weight, axis=-1)
        
        attn_weight = paddle.nn.functional.dropout(x=attn_weight, p=
            dropout_p, training=True)
        return attn_weight @ value

    def forward(self, x: paddle.Tensor, freqs_cis: paddle.Tensor=None,
        input_pos: Optional[paddle.Tensor]=None, mask: Optional[paddle.
        Tensor]=None):
        bsz, seqlen, _ = tuple(x.shape)
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        xq = xq.reshape([bsz, seqlen, self.n_head, self.head_dim])
        xk = xk.reshape([bsz, seqlen, self.n_kv_head, self.head_dim])
        xv = xv.reshape([bsz, seqlen, self.n_kv_head, self.head_dim])
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = map(lambda x: x.transpose(perm=paddle_aux.
            transpose_aux_func(x.ndim, 1, 2)), (xq, xk, xv))
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(repeats=self.n_head // self.n_kv_head,
            axis=1)
        values = values.repeat_interleave(repeats=self.n_head // self.
            n_kv_head, axis=1)
        # output = paddle.nn.functional.scaled_dot_product_attention(query=xq,
        #     key=keys, value=values, attn_mask=mask, is_causal=True if mask is
        #     None else False, dropout_p=self.attn_dropout_p if self.training
        #      else 0)
        
        output = self.scaled_dot_product_attention(query=xq,
            key=keys, value=values, attn_mask=mask, is_causal=True if mask is
            None else False, dropout_p=self.attn_dropout_p if self.training
             else 0)
         
        output = output.transpose(perm=paddle_aux.transpose_aux_func(output
            .ndim, 1, 2)).contiguous().reshape([bsz, seqlen, self.dim])
        output = self.resid_dropout(self.wo(output))
        return output


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int=10000,
    cls_token_num=120):
    half_dim = n_elem // 2
    freqs = 1.0 / base ** (paddle.arange(start=0, end=half_dim, step=2)[:
        half_dim // 2].astype(dtype='float32') / half_dim)
    t = paddle.arange(end=grid_size).astype(dtype='float32')
    freqs = paddle.outer(x=t, y=freqs)
    freqs_grid = paddle.concat(x=[freqs[:, None, :].expand(shape=[-1,
        grid_size, -1]), freqs[None, :, :].expand(shape=[grid_size, -1, -1]
        )], axis=-1)
    cache_grid = paddle.stack(x=[paddle.cos(x=freqs_grid), paddle.sin(x=
        freqs_grid)], axis=-1)
    cache = cache_grid.flatten(start_axis=0, stop_axis=1)
    cond_cache = paddle.concat(x=[paddle.zeros(shape=[cls_token_num, n_elem //
        2, 2]), cache])
    return cond_cache


def apply_rotary_emb(x: paddle.Tensor, freqs_cis: paddle.Tensor):
    xshaped = x.astype(dtype='float32').reshape([*tuple(x.shape)[:-1], -1, 2])
    freqs_cis = freqs_cis.reshape([1, xshaped.shape[1], 1, xshaped.shape[3], 2])
    freqs_cis = freqs_cis.to(xshaped.dtype)
    x_out2 = paddle.stack(x=[xshaped[..., 0] * freqs_cis[..., 0] - xshaped[
        ..., 1] * freqs_cis[..., 1], xshaped[..., 1] * freqs_cis[..., 0] + 
        xshaped[..., 0] * freqs_cis[..., 1]], axis=-1)
    x_out2 = x_out2.flatten(start_axis=3)
    return x_out2


class Block(paddle.nn.Layer):

    def __init__(self, config, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else paddle.nn.Identity()
      

    def forward(self, x: paddle.Tensor, freqs_cis: paddle.Tensor, start_pos:
        int, mask: Optional[paddle.Tensor]=None):
        h = x + self.drop_path(self.attention(self.attention_norm(x),
            freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class KVCache(paddle.nn.Layer):

    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype
        ):
        super().__init__()
        cache_shape = max_batch_size, n_head, max_seq_length, head_dim
        self.register_buffer(name='k_cache', tensor=paddle.zeros(shape=
            cache_shape, dtype=dtype))
        self.register_buffer(name='v_cache', tensor=paddle.zeros(shape=
            cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert tuple(input_pos.shape)[0] == tuple(k_val.shape)[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class GPT(paddle.nn.Layer):

    def __init__(self, vocab_size, block_size, spatial_n_layer=12, n_head=8,
        dim=256, cond_dim=256, factorized_n_layer=2, embd_pdrop=0.0,
        resid_dropout_p=0.0, attn_dropout_p=0.0, ffn_dropout_p=0.1,
        drop_path_rate=0.0, n_unmasked=0, token_factorization=False,
        max_batch_size=32, max_seq_len=2048, class_num=1000, token_drop=0.1,
        cls_token_num=1, rope_base=10000, norm_eps=1e-05,
        ffn_dim_multiplier=None, initalizer_range=0.02, multiple_of=256,
        n_kv_head=None, factorized_k=2, factorized_bits=[9, 9]):
        super().__init__()
        self.config = GPTConfig(vocab_size=vocab_size, block_size=
            block_size, cond_dim=cond_dim, embd_pdrop=embd_pdrop,
            resid_dropout_p=resid_dropout_p, attn_dropout_p=attn_dropout_p,
            spatial_n_layer=spatial_n_layer, factorized_n_layer=
            factorized_n_layer, n_head=n_head, dim=dim, ffn_dropout_p=
            ffn_dropout_p, drop_path_rate=drop_path_rate, n_unmasked=
            n_unmasked, token_factorization=token_factorization, class_num=
            class_num, token_drop=token_drop, cls_token_num=cls_token_num,
            rope_base=rope_base, norm_eps=norm_eps, ffn_dim_multiplier=
            ffn_dim_multiplier, initializer_range=initalizer_range,
            multiple_of=multiple_of, max_batch_size=max_batch_size,
            max_seq_len=max_seq_len, n_kv_head=n_kv_head, factorized_k=
            factorized_k, factorized_bits=factorized_bits)
        if token_factorization:
            self.pre_emb = paddle.nn.Embedding(num_embeddings=2 **
                factorized_bits[0], embedding_dim=self.config.dim)
            self.post_emb = paddle.nn.Embedding(num_embeddings=2 **
                factorized_bits[1], embedding_dim=self.config.dim)
            self.class_emb = LabelEmbedder(self.config.class_num, self.
                config.dim)
        else:
            self.tok_emb = paddle.nn.Embedding(num_embeddings=self.config.
                vocab_size, embedding_dim=self.config.dim)
            self.class_emb = LabelEmbedder(self.config.class_num, self.
                config.dim)
        self.token_drop = paddle.nn.Dropout(p=self.config.token_drop)
        spatial_dpr = [x.item() for x in paddle.linspace(start=0, stop=self
            .config.drop_path_rate, num=self.config.spatial_n_layer)]
        factorized_dpr = [x.item() for x in paddle.linspace(start=0, stop=
            self.config.drop_path_rate, num=self.config.factorized_n_layer)]
        self.spatial_blocks = paddle.nn.LayerList()
        for idx in range(self.config.spatial_n_layer):
            self.spatial_blocks.append(Block(self.config, spatial_dpr[idx]))
        self.factorized_blocks = paddle.nn.LayerList()
        for idx in range(self.config.factorized_n_layer):
            self.factorized_blocks.append(Block(self.config, factorized_dpr
                [idx]))
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.token_factorization = token_factorization
        assert token_factorization is True
        self.head = paddle.nn.LayerList(sublayers=[paddle.nn.Linear(
            in_features=self.config.dim, out_features=2 ** self.config.
            factorized_bits[_], bias_attr=False) for _ in range(factorized_k)])
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim //
            self.config.n_head, self.config.rope_base, self.config.
            cls_token_num)
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.initalize_weights()

    def initalize_weights(self):
        self.apply(self._init_weights)
        if self.token_factorization:
            for i in range(self.config.factorized_k):
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(self.head[i].weight)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, paddle.nn.Linear):
            normal_(module.weight.data, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias.data)
        elif isinstance(module, paddle.nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=std)


    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.spatial_blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length,
                self.config.n_head, head_dim, dtype)
        causal_mask = paddle.tril(x=paddle.ones(shape=[self.max_seq_length,
            self.max_seq_length], dtype='bool'))
        self.causal_mask = causal_mask.unsqueeze(axis=0).tile(repeat_times=
            [self.max_batch_size, 1, 1])
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim //
            self.config.n_head, self.config.rope_base, self.config.
            cls_token_num)

    def setup_factorized_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        self.max_batch_size = max_batch_size
        max_seq_length = find_multiple(max_seq_length, 8)
        for b in self.factorized_blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length,
                self.config.n_head, head_dim, dtype)

    def forward(self, idx, input_pos=None, mask=None, targets=None):
        if self.token_factorization:
            idx_pre, idx_post, idx_cls = idx[0], idx[1], idx[2]
            token_embeddings_pre = self.pre_emb(idx_pre)
            token_embeddings_post = self.post_emb(idx_post)
            cls_token_embeddings = self.class_emb(idx_cls, train=self.training
                )[:, :self.config.cls_token_num]
            token_embeddings = token_embeddings_pre + token_embeddings_post
            token_embeddings = paddle.concat(x=[cls_token_embeddings,
                token_embeddings[:, :-1, :]], axis=1)
            h = self.token_drop(token_embeddings)
        else:
            idx, idx_cls = idx[0], idx[1]
            token_embeddings = self.tok_emb(idx)
            cls_token_embeddings = self.class_emb(idx_cls, train=self.training
                )[:, :self.config.cls_token_num]
            token_embeddings = paddle.concat(x=[cls_token_embeddings,
                token_embeddings], axis=1)
            h = self.token_drop(token_embeddings)
        B, N, D = tuple(h.shape)
        freqs_cis = self.freqs_cis[:tuple(token_embeddings.shape)[1]]
        freqs_cis = freqs_cis.to(h.place)
        for block in self.spatial_blocks:
            h = block(h, freqs_cis, input_pos, mask)
        assert self.token_factorization
        token_embeddings_pre = self.pre_emb(idx_pre)
        token_embeddings_post = self.post_emb(idx_post)
        factorized_ctx = paddle.stack(x=[h, token_embeddings_pre], axis=-2)
        if not True:
            factorized_ctx = factorized_ctx.contiguous()
        factorized_ctx = factorized_ctx.reshape([B * N, -1, D])
        factorized_ctx_freqs_cis = freqs_cis[:tuple(factorized_ctx.shape)[1]]
        for block in self.factorized_blocks:
            factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis,
                mask)
        if not True:
            factorized_ctx = factorized_ctx.contiguous()
        h = factorized_ctx.reshape([B, N, -1, D])
        h = self.norm(h)
        logits = [self.head[i](h[:, :, i, :]) for i in range(self.config.
            factorized_k)]
        loss = None
        if targets is not None:
            loss = paddle.nn.functional.cross_entropy(input=logits.reshape([-1,
                logits.shape[-1]]), label=targets.reshape([-1]))
        return logits, loss

    def generate_context(self, idx, input_pos=None, targets=None,
        first_step=False):
        """
        Generate context token for Inference
        """
        assert not self.training
        if first_step:
            token_embeddings = self.class_emb(idx, train=self.training)
        else:
            idx_pre, idx_post = idx[0], idx[1]
            token_embedding_pre = self.pre_emb(idx_pre)
            token_embedding_post = self.post_emb(idx_post)
            token_embeddings = token_embedding_pre + token_embedding_post
        bs, N, D = tuple(token_embeddings.shape)
        
        dtype = self.causal_mask.dtype
        mask = paddle.index_select(self.causal_mask[:bs,None].astype("int"),input_pos,axis=2).astype(dtype)
        h = self.token_drop(token_embeddings)
        freq_cis = self.freqs_cis[input_pos]
        for block in self.spatial_blocks:
            h = block(h, freq_cis, input_pos, mask)
        return h

    def decode_subtoken(self, h, x, input_pos=None, first_step=False):
        """
        Auto-Regressive generate subtoken
        """
        B, N, D = tuple(h.shape)
        if not True:
            h = h.contiguous()
        if first_step:
            factorized_ctx = h.reshape(B * N, -1, D)
        else:
            idx = x[0]
            token_embedding = self.pre_emb(idx)
            factorized_ctx = token_embedding.reshape(B * N, -1, D)
        # mask = self.causal_mask[:B, None, input_pos]
        dtype = self.causal_mask.dtype
        mask = paddle.index_select(self.causal_mask[:B,None].astype("int"),input_pos,axis=2).astype(dtype)
        factorized_ctx_freqs_cis = self.freqs_cis[input_pos]
        factorized_ctx_freqs_cis = factorized_ctx_freqs_cis.to(h.place)
        for block in self.factorized_blocks:
            factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis,
                start_pos=input_pos, mask=mask)
        h = factorized_ctx.reshape(B, N, -1, D)
        h = self.norm(h)
        logits = self.head[0](h[:, :, 0, :]) if first_step else self.head[1](h
            [:, :, 0, :])
        return logits


@paddle.no_grad()
def sample(x, model, steps, temperature=1.0, sample_logits=True, top_k=None,
    top_p=None, callback=None, token_factorization=True, cfg_scale=1.0):
    assert token_factorization is True
    k = 2
    bs, _ = tuple(x.shape)

    if cfg_scale[0] > 1.0:
        cond_token, uncond_token = paddle_aux.split(x=x, num_or_sections=bs //
            2, axis=0)
        sample_pre, sample_post = cond_token, cond_token
    else:
        cond_token = x
        sample_pre, sample_post = cond_token, cond_token
    cond_len = tuple(x.shape)[1]
    if cfg_scale[0] > 1.0:
        max_batch_size = tuple(x.shape)[0] // 2
    else:
        max_batch_size = tuple(x.shape)[0]
    max_seq_length = cond_len + steps

    max_batch_size_cfg = max_batch_size * 2 if cfg_scale[0
        ] >= 1.0 else max_batch_size
    model.setup_caches(max_batch_size=max_batch_size_cfg,
        max_seq_length=max_seq_length, dtype=model.class_emb.
        embedding_table.weight.dtype)
    
    for n in range(steps):
        if n == 0:
            input_pos = paddle.arange(start=0, end=cond_len)
        elif n == 1:
            input_pos = paddle.to_tensor(data=[cond_len])
        else:
            input_pos = input_pos + 1
        h = model.generate_context(x, input_pos=input_pos, first_step=n == 0)
        x = []
      
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale[0
            ] >= 1.0 else max_batch_size
        model.setup_factorized_caches(max_batch_size=max_batch_size_cfg,
            max_seq_length=max_seq_length, dtype=model.class_emb.
            embedding_table.weight.dtype)

        for i in range(k):
            if i == 0:
                if cfg_scale[i] > 1.0:
                    factor_x = paddle.concat(x=[cond_token, uncond_token])
                else:
                    factor_x = cond_token
            factor_input_pos = paddle.to_tensor(data=[i])
            
            logits = model.decode_subtoken(h, factor_x, factor_input_pos,
                first_step=i == 0)
          
            if cfg_scale[i] > 1.0:
                cond_logits, uncond_logits = paddle_aux.split(x=logits,
                    num_or_sections=bs // 2, axis=0)
                logits = uncond_logits + (cond_logits - uncond_logits
                    ) * cfg_scale[i]
            factor_x = sample_from_logits(logits, temperature[i], top_k[i],
                top_p[i])
           
            if i == 0:
                sample_pre = paddle.concat(x=(sample_pre, factor_x), axis=1)
            else:
                sample_post = paddle.concat(x=(sample_post, factor_x), axis=1)
            if cfg_scale[i] > 1.0:
                cfg_x = paddle.concat(x=[factor_x, factor_x])
                factor_x = [cfg_x, paddle.concat(x=[cond_token, uncond_token])]
                x.append(cfg_x)
            else:
                non_cfg_x = factor_x
                factor_x = non_cfg_x, cond_token
                x.append(non_cfg_x)
        if cfg_scale[0] > 1.0:
            x.append(paddle.concat(x=[cond_token, uncond_token]))
        else:
            x.append(cond_token)
    sample_pre = sample_pre[:, cond_len:]
    sample_post = sample_post[:, cond_len:]
    sample = (sample_pre, sample_post)
    return sample


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None,
    sample_logits=True):
    logits = logits[:, -1, :] / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    sample_logits=False
    if not sample_logits:
        _, x = paddle.topk(probs, k=1, axis=-1)
    else:
        x = paddle.multinomial(x=probs, num_samples=1)
    return x
