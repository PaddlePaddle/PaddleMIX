import sys
sys.path.append(
    './utils')
import paddle_aux
import paddle
import numpy as np
from einops import rearrange


class VectorQuantizer(paddle.nn.Layer):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = paddle.nn.Embedding(num_embeddings=self.n_e,
            embedding_dim=self.e_dim)
        self.embedding.weight.data.uniform_(min=-1.0 / self.n_e, max=1.0 /
            self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        z = z.transpose(perm=[0, 2, 3, 1]).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = paddle.sum(x=z_flattened ** 2, axis=1, keepdim=True) + paddle.sum(x
            =self.embedding.weight ** 2, axis=1) - 2 * paddle.matmul(x=
            z_flattened, y=self.embedding.weight.t())
        min_encoding_indices = paddle.argmin(x=d, axis=1).unsqueeze(axis=1)
        min_encodings = paddle.zeros(shape=[tuple(min_encoding_indices.
            shape)[0], self.n_e]).to(z)
        min_encodings.put_along_axis_(axis=1, indices=min_encoding_indices,
            values=1, broadcast=False)
        z_q = paddle.matmul(x=min_encodings, y=self.embedding.weight).view(
            tuple(z.shape))
        loss = paddle.mean(x=(z_q.detach() - z) ** 2
            ) + self.beta * paddle.mean(x=(z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        e_mean = paddle.mean(x=min_encodings, axis=0)
        perplexity = paddle.exp(x=-paddle.sum(x=e_mean * paddle.log(x=
            e_mean + 1e-10)))
        z_q = z_q.transpose(perm=[0, 3, 1, 2]).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        min_encodings = paddle.zeros(shape=[tuple(indices.shape)[0], self.n_e]
            ).to(indices)
        min_encodings.put_along_axis_(axis=1, indices=indices[:, None],
            values=1, broadcast=False)
        z_q = paddle.matmul(x=min_encodings.astype(dtype='float32'), y=self
            .embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.transpose(perm=[0, 3, 1, 2]).contiguous()
        return z_q


class GumbelQuantize(paddle.nn.Layer):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed,
        straight_through=True, kl_weight=0.0005, temp_init=1.0,
        use_vqinterface=True, remap=None, unknown_index='random'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = paddle.nn.Conv2D(in_channels=num_hiddens, out_channels=
            n_embed, kernel_size=1)
        self.embed = paddle.nn.Embedding(num_embeddings=n_embed,
            embedding_dim=embedding_dim)
        self.use_vqinterface = use_vqinterface
        self.remap = remap
        if self.remap is not None:
            self.register_buffer(name='used', tensor=paddle.to_tensor(data=
                np.load(self.remap)))
            self.re_embed = tuple(self.used.shape)[0]
            self.unknown_index = unknown_index
            if self.unknown_index == 'extra':
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f'Remapping {self.n_embed} indices to {self.re_embed} indices. Using {self.unknown_index} for unknown indices.'
                )
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = tuple(inds.shape)
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).astype(dtype=
            'int64')
        new = match.argmax(axis=-1)
        unknown = match.sum(axis=2) < 1
        if self.unknown_index == 'random':
            new[unknown] = paddle.randint(low=0, high=self.re_embed, shape=
                tuple(new[unknown].shape)).to(device=new.place)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = tuple(inds.shape)
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > tuple(self.used.shape)[0]:
            inds[inds >= tuple(self.used.shape)[0]] = 0
        back = paddle.take_along_axis(arr=used[None, :][tuple(inds.shape)[0
            ] * [0], :], axis=1, indices=inds, broadcast=False)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        logits = self.proj(z)
        if self.remap is not None:
            full_zeros = paddle.zeros_like(x=logits)
            logits = logits[:, self.used, ...]
        soft_one_hot = paddle.nn.functional.gumbel_softmax(x=logits,
            temperature=temp, axis=1, hard=hard)
        if self.remap is not None:
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = paddle.einsum('b n h w, n d -> b d h w', soft_one_hot, self.
            embed.weight)
        qy = paddle.nn.functional.softmax(x=logits, axis=1)
        diff = self.kl_weight * paddle.sum(x=qy * paddle.log(x=qy * self.
            n_embed + 1e-10), axis=1).mean()
        ind = soft_one_hot.argmax(axis=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b * h * w == tuple(indices.shape)[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = paddle.nn.functional.one_hot(num_classes=self.n_embed, x=
            indices).astype('int64').transpose(perm=[0, 3, 1, 2]).astype(dtype
            ='float32')
        z_q = paddle.einsum('b n h w, n d -> b d h w', one_hot, self.embed.
            weight)
        return z_q


class VectorQuantizer2(paddle.nn.Layer):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index='random',
        sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding = paddle.nn.Embedding(num_embeddings=self.n_e,
            embedding_dim=self.e_dim)
        self.embedding.weight.data.uniform_(min=-1.0 / self.n_e, max=1.0 /
            self.n_e)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer(name='used', tensor=paddle.to_tensor(data=
                np.load(self.remap)))
            self.re_embed = tuple(self.used.shape)[0]
            self.unknown_index = unknown_index
            if self.unknown_index == 'extra':
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f'Remapping {self.n_e} indices to {self.re_embed} indices. Using {self.unknown_index} for unknown indices.'
                )
        else:
            self.re_embed = n_e
        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = tuple(inds.shape)
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).astype(dtype=
            'int64')
        new = match.argmax(axis=-1)
        unknown = match.sum(axis=2) < 1
        if self.unknown_index == 'random':
            new[unknown] = paddle.randint(low=0, high=self.re_embed, shape=
                tuple(new[unknown].shape)).to(device=new.place)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = tuple(inds.shape)
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > tuple(self.used.shape)[0]:
            inds[inds >= tuple(self.used.shape)[0]] = 0
        back = paddle.take_along_axis(arr=used[None, :][tuple(inds.shape)[0
            ] * [0], :], axis=1, indices=inds, broadcast=False)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, 'Only for interface compatible with Gumbel'
        assert rescale_logits == False, 'Only for interface compatible with Gumbel'
        assert return_logits == False, 'Only for interface compatible with Gumbel'
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = paddle.sum(x=z_flattened ** 2, axis=1, keepdim=True) + paddle.sum(x
            =self.embedding.weight ** 2, axis=1) - 2 * paddle.einsum(
            'bd,dn->bn', z_flattened, rearrange(self.embedding.weight,
            'n d -> d n'))
        min_encoding_indices = paddle.argmin(x=d, axis=1)
        z_q = self.embedding(min_encoding_indices).view(tuple(z.shape))
        perplexity = None
        min_encodings = None
        if not self.legacy:
            loss = self.beta * paddle.mean(x=(z_q.detach() - z) ** 2
                ) + paddle.mean(x=(z_q - z.detach()) ** 2)
        else:
            loss = paddle.mean(x=(z_q.detach() - z) ** 2
                ) + self.beta * paddle.mean(x=(z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(tuple(z.
                shape)[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(tuple(z_q.
                shape)[0], tuple(z_q.shape)[2], tuple(z_q.shape)[3])
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.transpose(perm=[0, 3, 1, 2]).contiguous()
        return z_q


class EmbeddingEMA(paddle.nn.Layer):

    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-05):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = paddle.randn(shape=[num_tokens, codebook_dim])
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =weight, trainable=False)
        self.cluster_size = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=num_tokens), trainable=False)
        self.embed_avg = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=weight.clone(), trainable=False)
        self.update = True

    def forward(self, embed_id):
        return paddle.nn.functional.embedding(x=embed_id, weight=self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.multiply_(y=paddle.to_tensor(self.decay)).add_(y
            =paddle.to_tensor((1 - self.decay) * new_cluster_size))

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.multiply_(y=paddle.to_tensor(self.decay)).add_(y
            =paddle.to_tensor((1 - self.decay) * new_embed_avg))

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + 
            num_tokens * self.eps) * n
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(
            axis=1)
        paddle.assign(embed_normalized, output=self.weight.data)


class EMAVectorQuantizer(paddle.nn.Layer):

    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-05,
        remap=None, unknown_index='random'):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim,
            decay, eps)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer(name='used', tensor=paddle.to_tensor(data=
                np.load(self.remap)))
            self.re_embed = tuple(self.used.shape)[0]
            self.unknown_index = unknown_index
            if self.unknown_index == 'extra':
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f'Remapping {self.n_embed} indices to {self.re_embed} indices. Using {self.unknown_index} for unknown indices.'
                )
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = tuple(inds.shape)
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).astype(dtype=
            'int64')
        new = match.argmax(axis=-1)
        unknown = match.sum(axis=2) < 1
        if self.unknown_index == 'random':
            new[unknown] = paddle.randint(low=0, high=self.re_embed, shape=
                tuple(new[unknown].shape)).to(device=new.place)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = tuple(inds.shape)
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > tuple(self.used.shape)[0]:
            inds[inds >= tuple(self.used.shape)[0]] = 0
        back = paddle.take_along_axis(arr=used[None, :][tuple(inds.shape)[0
            ] * [0], :], axis=1, indices=inds, broadcast=False)
        return back.reshape(ishape)

    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        d = z_flattened.pow(y=2).sum(axis=1, keepdim=True
            ) + self.embedding.weight.pow(y=2).sum(axis=1) - 2 * paddle.einsum(
            'bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = paddle.argmin(x=d, axis=1)
        z_q = self.embedding(encoding_indices).view(tuple(z.shape))
        encodings = paddle.nn.functional.one_hot(num_classes=self.
            num_tokens, x=encoding_indices).astype('int64').astype(z.dtype)
        avg_probs = paddle.mean(x=encodings, axis=0)
        perplexity = paddle.exp(x=-paddle.sum(x=avg_probs * paddle.log(x=
            avg_probs + 1e-10)))
        if self.training and self.embedding.update:
            encodings_sum = encodings.sum(axis=0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            embed_sum = encodings.transpose(perm=paddle_aux.
                transpose_aux_func(encodings.ndim, 0, 1)) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)
        loss = self.beta * paddle.nn.functional.mse_loss(input=z_q.detach(),
            label=z)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)
