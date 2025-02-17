import paddle
"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.

Refer to 
https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py
https://github.com/theAdamColton/ijepa-enhanced/blob/7edef5f7288ae8f537f0db8a10044a2a487f70c9/ijepa_enhanced/lfq.py
"""
from math import log2, ceil
from collections import namedtuple
from einops import rearrange, reduce, pack, unpack
LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy',
    'codebook_entropy', 'commitment', 'avg_probs'])


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def entropy(prob):
    return (-prob * paddle.log(x=prob + 1e-05)).sum(axis=-1)


def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(axis=-1)
    return x * y


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(axis=tuple(range(m.ndim)))


def entropy_loss(logits, mask=None, temperature=0.01,
    sample_minimization_weight=1.0, batch_maximization_weight=1.0, eps=1e-05):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = paddle.nn.functional.softmax(x=logits / temperature, axis=-1)
    log_probs = paddle.nn.functional.log_softmax(x=logits / temperature +
        eps, axis=-1)
    if mask is not None:
        avg_probs = masked_mean(probs, mask)
    else:
        avg_probs = reduce(probs, '... D -> D', 'mean')
    avg_entropy = -paddle.sum(x=avg_probs * paddle.log(x=avg_probs + eps))
    sample_entropy = -paddle.sum(x=probs * log_probs, axis=-1)
    if mask is not None:
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = paddle.mean(x=sample_entropy)
    loss = (sample_minimization_weight * sample_entropy - 
        batch_maximization_weight * avg_entropy)
    return sample_entropy, avg_entropy, loss


class LFQ(paddle.nn.Layer):

    def __init__(
        self, 
        *, 
        dim=None, 
        codebook_size=None, 
        num_codebooks=1,
        sample_minimization_weight=1.0, 
        batch_maximization_weight=1.0,
        token_factorization=False, 
        factorized_bits=[9, 9]):
        super().__init__()
        
        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'
        
        self.codebook_size = default(codebook_size, lambda : 2 ** dim)
        self.codebook_dim = int(log2(codebook_size))
        
        codebook_dims = self.codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        
        has_projections = dim != codebook_dims
        self.has_projections = has_projections
       
        self.dim = dim
        self.codebook_dim = self.codebook_dim
        self.num_codebooks = num_codebooks
        
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        
        self.token_factorization = token_factorization
        if not self.token_factorization:
            self.register_buffer(name='mask', tensor=2 ** paddle.arange(end=self.codebook_dim), persistable=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer(name='pre_mask', tensor=2 ** paddle.arange(end=factorized_bits[0]), persistable=False)
            self.register_buffer(name='post_mask', tensor=2 ** paddle.arange(end=factorized_bits[1]), persistable=False)
            
        self.register_buffer(name='zero', tensor=paddle.to_tensor(data=0.0),persistable=False)
        
        all_codes = paddle.arange(end=codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0
        
        self.register_buffer(name='codebook', tensor=codebook, persistable=False)

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** paddle.arange(dtype='int64', end=self.codebook_dim)
        x = x.unsqueeze(axis=-1) & mask != 0
        return x

    def get_codebook_entry(self, x, bhwc, order):
        if self.token_factorization:
            if order == 'pre':
                mask = 2 ** paddle.arange(dtype='int64', end=self.
                    factorized_bits[0])
            else:
                mask = 2 ** paddle.arange(dtype='int64', end=self.
                    factorized_bits[1])
        else:
            mask = 2 ** paddle.arange(dtype='int64', end=self.codebook_dim)
        x = x.unsqueeze(axis=-1) & mask != 0
        x = x * 2.0 - 1.0
        b, h, w, c = bhwc
        x = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w, c=c)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert tuple(bits.shape)[-1] == self.codebook_dim
        indices = 2 ** paddle.arange(start=0, end=self.codebook_dim, step=1,
            dtype='int64')
        return (bits * indices).sum(axis=-1)

    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        x = x.to(self.dtype)
        x = x * 2 - 1
        x = rearrange(x, '... NC Z-> ... (NC Z)')
        return x

    def forward(self, x, inv_temperature=100.0, return_loss_breakdown=False,
        mask=None, return_loss=True):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        x = rearrange(x, 'b d ... -> b ... d')
        x, ps = pack_one(x, 'b * d')
        
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)
        
        codebook_value = paddle.to_tensor(data=[1.0], dtype='float32').to(dtype=x.dtype)
        quantized = paddle.where(condition=x > 0, x=codebook_value, y=-codebook_value)
        
        if self.token_factorization:
            indices_pre = reduce((quantized[..., :self.factorized_bits[0]] >
                0).astype(dtype='int32') * self.pre_mask.astype(dtype=
                'int32'), 'b n c d -> b n c', 'sum')
            indices_post = reduce((quantized[..., self.factorized_bits[0]:] >
                0).astype(dtype='int32') * self.post_mask.astype(dtype=
                'int32'), 'b n c d -> b n c', 'sum')
        else:
            indices = reduce((quantized > 0).astype(dtype='int32') * self.
                mask.astype(dtype='int32'), 'b n c d -> b n c', 'sum')
        if self.training and return_loss:
            logits = 2 * paddle.einsum('... i d, j d -> ... i j', x, self.codebook)
            
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits=logits, sample_minimization_weight=self.sample_minimization_weight, 
                batch_maximization_weight=self.batch_maximization_weight)
            
            avg_probs = self.zero
            
        else:
            per_sample_entropy = codebook_entropy = self.zero
            entropy_aux_loss = self.zero
            avg_probs = self.zero
            
        if self.training:
            commit_loss = paddle.nn.functional.mse_loss(input=x, label=quantized.detach(), reduction='none')
            
            if exists(mask):
                commit_loss = commit_loss[mask]
                
            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero
            
        quantized = x + (quantized - x).detach()
        quantized = rearrange(quantized, 'b n c d -> b n (c d)')
        quantized = unpack_one(quantized, ps, 'b * d')
        quantized = rearrange(quantized, 'b ... d -> b d ...')
        if self.token_factorization:
            indices_pre = unpack_one(indices_pre, ps, 'b * c')
            indices_post = unpack_one(indices_post, ps, 'b * c')
            indices_pre = indices_pre.flatten()
            indices_post = indices_post.flatten()
            indices = indices_pre, indices_post
        else:
            indices = unpack_one(indices, ps, 'b * c')
            indices = indices.flatten()
        ret = quantized, entropy_aux_loss, indices
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(per_sample_entropy, codebook_entropy,
            commit_loss, avg_probs)


if __name__ == '__main__':
    quantizer = LFQ(codebook_size=2 ** 18, dim=18,
        sample_minimization_weight=1.0, batch_maximization_weight=1.0)
    image_feats = paddle.randn(shape=[2, 18, 16, 16])
    quantized, indices, entropy_aux_loss = quantizer(image_feats,
        inv_temperature=100.0)
    assert tuple(image_feats.shape) == tuple(quantized.shape)
    assert (quantized == quantizer.indices_to_codes(indices)).astype('bool'
        ).all()
