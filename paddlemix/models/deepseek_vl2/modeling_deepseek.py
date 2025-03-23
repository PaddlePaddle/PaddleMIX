# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Paddle DeepSeek model and compatible with both DeepSeekV2 and DeepSeekV3"""
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
import paddlenlp
from paddle.distributed.fleet.utils import recompute
from paddlenlp.transformers import PretrainedModel
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.llama.modeling import LlamaAttention
from paddlenlp.utils.tools import get_env_device

from .configuration_deepseek import DeepseekV2Config
from paddlemix.models.qwen2_vl.bert_padding import index_first_axis, pad_input, unpad_input
from paddlemix.models.flash_attn_utils import (
    has_flash_attn_func,
)
flash_attn_func, flash_attn_varlen_func = has_flash_attn_func() # flash_attention, flash_attn_varlen_func
_IS_NPU = "npu" in paddle.get_device()

from ppdiffusers.utils import logging
logger = logging.get_logger(__name__)


class DeepseekV2RMSNorm(nn.Layer):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, eps=1e-6, use_sequence_parallel=True):
        """DeepseekV2RMSNorm is equivalent to T5LayerNorm

        Args:
            config (DeepseekV2Config): config dict of DeepseekV2
            hidden_size (_type_): history_states size
            eps (_type_, optional): eps value. Defaults to 1e-6.
            use_sequence_parallel (bool, optional): A switch to disable sequence parallelism for inputs that are not in tensor parallel mode.
                                                    By default, this is set to True.
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.variance_epsilon = eps

        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, hidden_states):
        if self.config.use_fused_rms_norm and get_env_device() == "xpu":
            if self.weight.dtype != hidden_states.dtype:
                hidden_states = paddle.cast(hidden_states, self.weight.dtype)
            try:
                import paddle_xpu_nn  # noqa: F821

                return paddle_xpu_nn.xpu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            except ImportError:
                raise NotImplementedError(
                    f"Implementation of fused_rms_norm is not available on {get_env_device()}. Please install paddle_xpu to use this feature"
                )

        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                hidden_states = hidden_states.astype("float32")
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        else:
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class DeepseekV2RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (
            paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32").to(device) / self.dim
        )

        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.place, dtype=paddle.get_default_dtype()
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.outer(x=t, y=self.inv_freq.to(t.place))
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().astype(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().astype(dtype), persistable=False)

    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.place, dtype=x.dtype)
        return self.cos_cached[:seq_len].astype(dtype=x.dtype), self.sin_cached[:seq_len].astype(dtype=x.dtype)


class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        t = t / self.scaling_factor
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().astype(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().astype(dtype), persistable=False)


class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (
                paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32").to(device) / self.dim
            )
            self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().astype(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().astype(dtype), persistable=False)


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (paddle.arange(dim, dtype=paddle.float32) - min) / (max - min)
    ramp_func = paddle.clip(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        self.inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = paddle.arange(seq_len, dtype=paddle.float32)

        freqs = paddle.outer(t, self.inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = paddle.concat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos() * _mscale
        self.sin_cached = emb.sin() * _mscale


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tuple(x.shape)[-1] // 2]
    x2 = x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(axis=unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(axis=unsqueeze_dim)
    b, h, s, d = tuple(q.shape)
    q = q.reshape([b, h, s, d // 2, 2]).transpose(perm=[0, 1, 2, 4, 3]).reshape([b, h, s, d])

    b, h, s, d = tuple(k.shape)
    k = k.reshape([b, h, s, d // 2, 2]).transpose(perm=[0, 1, 2, 4, 3]).reshape([b, h, s, d])

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class DeepseekV2MLP(nn.Layer):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None, is_moe=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=(self.gating_dim, self.n_routed_experts))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[self.n_routed_experts])
            )

    #     self.reset_parameters()

    # def reset_parameters(self) -> None:
    #     init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
    #         negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
    #     )
    #     init_KaimingUniform(self.weight)

    def forward(self, hidden_states):
        bsz, seq_len, h = tuple(hidden_states.shape)
        hidden_states = hidden_states.reshape([-1, h])
        logits = paddle.nn.functional.linear(
            x=hidden_states.astype("float32"), weight=self.weight.astype("float32"), bias=None
        )
        if self.scoring_func == "softmax":
            scores = paddle.nn.functional.softmax(logits, axis=-1, dtype="float32")
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")
        if self.topk_method == "greedy":
            topk_weight, topk_idx = paddle.topk(k=self.top_k, sorted=False, x=scores, axis=-1)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.reshape(bsz * seq_len, self.n_group, -1).max(dim=-1).values

            group_idx = paddle.topk(k=self.topk_group, sorted=False, x=group_scores, axis=-1)[1]
            group_mask = paddle.zeros_like(x=group_scores)
            group_mask.put_along_axis_(axis=1, indices=group_idx, values=1, broadcast=False)
            score_mask = (
                group_mask.unsqueeze(axis=-1)
                .expand(shape=[bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group])
                .reshape([bsz * seq_len, -1])
            )
            tmp_scores = scores.masked_fill(mask=~score_mask.astype(dtype="bool"), value=0.0)
            topk_weight, topk_idx = paddle.topk(k=self.top_k, sorted=False, x=tmp_scores, axis=-1)
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.reshape([bsz * seq_len, -1]) + self.e_score_correction_bias.unsqueeze(axis=0)
            group_scores = scores_for_choice.reshape([bsz * seq_len, self.n_group, -1]).topk(k=2, axis=-1)[0].sum(axis=-1)

            group_idx = paddle.topk(k=self.topk_group, sorted=False, x=group_scores, axis=-1)[1]
            group_mask = paddle.zeros_like(x=group_scores)
            group_mask.put_along_axis_(axis=1, indices=group_idx, values=1, broadcast=False)
            # todo
            score_mask = (
                group_mask.unsqueeze(axis=-1)
                .expand(shape=[bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group])
                .reshape([bsz * seq_len, -1])
            )
            tmp_scores = scores_for_choice.masked_fill(mask=~score_mask.astype(dtype="bool"), value=0.0)
            _, topk_idx = paddle.topk(k=self.top_k, sorted=False, x=tmp_scores, axis=-1)
            topk_weight = scores.take_along_axis(axis=1, indices=topk_idx, broadcast=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(axis=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.reshape([bsz, -1])
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.reshape([bsz, seq_len, -1])
                ce = paddle.zeros(shape=[bsz, self.n_routed_experts])
                ce.put_along_axis_(
                    axis=1,
                    indices=topk_idx_for_aux_loss,
                    values=paddle.ones(shape=[bsz, seq_len * aux_topk]),
                    reduce="add",
                ).divide_(y=paddle.to_tensor(seq_len * aux_topk / self.n_routed_experts))
                aux_loss = (ce * scores_for_seq_aux.mean(axis=1)).sum(axis=1).mean() * self.alpha
            else:
                mask_ce = paddle.nn.functional.one_hot(
                    num_classes=self.n_routed_experts, x=topk_idx_for_aux_loss.reshape([-1])
                ).astype("int64")
                ce = mask_ce.astype(dtype="float32").mean(axis=0)
                Pi = scores_for_aux.mean(axis=0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert paddle.numel(loss) == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = paddle.ones(1, dtype=ctx.dtype)
        return grad_output, grad_loss


# from paddlenlp.transformers.deepseek_v2.modeling import DeepseekV2MoE # diff
class DeepseekV2MoE(paddle.nn.Layer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV2MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.LayerList(
                [
                    DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size)
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
        flat_topk_idx = topk_idx.reshape([-1])
        # remove the infer method
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, axis=0)
            y = paddle.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                # y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
                if paddle.any(flat_topk_idx == i):
                    y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])

            y = (y.reshape([*topk_weight.shape, -1]) * topk_weight.unsqueeze(-1)).sum(axis=1)
            y = paddle.cast(y, hidden_states.dtype).reshape([*orig_shape])
            if self.gate.alpha > 0.0:
                y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).reshape([*orig_shape])
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @paddle.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = paddle.zeros(shape=(tuple(topk_ids.shape)[0], len(self.experts)), dtype=topk_ids.dtype)
        cnts.put_along_axis_(axis=1, indices=topk_ids, values=1, broadcast=False)
        tokens_per_expert = cnts.sum(axis=0)
        idxs = topk_ids.reshape([-1]).argsort()
        sorted_tokens = x[idxs // tuple(topk_ids.shape)[1]]
        sorted_tokens_shape = tuple(sorted_tokens.shape)
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.reshape([self.ep_size, -1]).sum(axis=1)
            tokens_per_expert_group = paddle.empty(
                shape=tuple(tokens_per_expert.shape)[0], dtype=tokens_per_expert.dtype
            )
            paddle.distributed.alltoall_single(tokens_per_expert_group, tokens_per_expert)

            output_splits = tokens_per_expert_group.reshape([self.ep_size, -1]).sum(axis=1).cpu().numpy().tolist()
            gathered_tokens = paddle.empty(
                shape=[tokens_per_expert_group.sum(axis=0).cpu().item(), tuple(sorted_tokens.shape)[1]],
                dtype=sorted_tokens.dtype,
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            paddle.distributed.alltoall(
                out_tensor_list=list(gathered_tokens.split(output_splits)),
                in_tensor_list=list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.reshape([self.ep_size, self.experts_per_rank]).sum(
                axis=0
            )
            gatherd_idxs = np.zeros(shape=(tuple(gathered_tokens.shape)[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = paddle.concat(x=outputs, axis=0) if len(outputs) else paddle.empty(shape=[0], dtype=sorted_tokens.dtype)

        if self.ep_size > 1:
            new_x = paddle.empty_like(x=outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = paddle.empty(shape=sorted_tokens_shape, dtype=new_x.dtype)
            paddle.distributed.alltoall(
                out_tensor_list=list(gathered_tokens.split(input_split_sizes)),
                in_tensor_list=list(new_x.split(output_splits)),
            )
            outs = gathered_tokens
        new_x = paddle.empty_like(x=outs)
        new_x[idxs] = outs
        final_out = (
            new_x.reshape([*tuple(topk_ids.shape), -1])
            .astype(topk_weight.dtype)
            .multiply_(y=paddle.to_tensor(topk_weight.unsqueeze(axis=-1)))
            .sum(axis=1)
            .astype(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(paddle.nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True
        self.fuse_rope = config.use_fused_rope

        if config.num_nextn_predict_layers > 0:
            self.seq_length = config.seq_length - config.num_nextn_predict_layers
        else:
            self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        self.enable_recompute = False
        # self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias_attr=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias_attr=config.attention_bias)
            self.q_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias_attr=False)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias_attr=config.attention_bias)
        self.kv_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias_attr=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias_attr=config.attention_bias)

        self._init_rope()

        self.softmax_scale = self.q_head_dim**-0.5
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        # self.attn_func = scaled_dot_product_attention

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.v_head_dim]).transpose([1, 0, 2, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:

        bsz, q_len, _ = tuple(hidden_states.shape)
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.reshape([bsz, q_len, self.num_heads, self.q_head_dim]).transpose(perm=[0, 2, 1, 3])
        q_nope, q_pe = paddle.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.reshape([bsz, q_len, 1, self.qk_rope_head_dim]).transpose(perm=[0, 2, 1, 3])

        kv_seq_len = tuple(k_pe.shape)[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1] # [1, 1304, 1, 64]
        cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        # [1, 16, 2035, 64] [1, 1, 2035, 64] [2035, 64] [2035, 64]  [1, 2035]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        # [1, 16, 1, 64] [1, 1, 1, 64]

        if use_cache and past_key_value is not None:
            compressed_kv = compressed_kv.unsqueeze(axis=2)
            k_pe = k_pe.transpose(perm=[0, 2, 1, 3])  # (b h l d) to (b l h d)
            k_pe = paddle.concat([past_key_value[0], k_pe], axis=1)
            compressed_kv = paddle.concat([past_key_value[1], compressed_kv], axis=1)

            past_key_value = (k_pe, compressed_kv)

            k_pe = k_pe.transpose(perm=[0, 2, 1, 3])  # go back to (b l h d)
            compressed_kv = compressed_kv.squeeze(2)
        elif use_cache:
            past_key_value = (k_pe.transpose([0, 2, 1, 3]), compressed_kv.unsqueeze(axis=2))
        else:
            past_key_value = None

        # shit tranpose liner weight
        kv_b_proj = self.kv_b_proj.weight.T.reshape([self.num_heads, -1, self.kv_lora_rank]) # [512, 4096] -> [16, -1, 512]
        q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :] # [16, 128, 512]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :] # [16, 128, 512]

        q_nope = paddle.matmul(q_nope, q_absorb) # [1, 16, 1304, 512]
        attn_weights = (
            paddle.matmul(q_pe, k_pe.transpose([0, 1, 3, 2])) # [1, 16, 1304, 64] * [1, 1, 1304, 64]
            + paddle.matmul(q_nope, compressed_kv.unsqueeze(axis=-3).transpose([0, 1, 3, 2])) #  [1, 16, 1304, 512] * [1, 1, 1304, 512]
        ) * self.softmax_scale

        if tuple(attn_weights.shape) != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {tuple(attn_weights.shape)}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {tuple(attention_mask.shape)}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").to(q_pe.dtype)
        attn_weights = F.dropout(attn_weights, self.attention_dropout, training=self.training)
        attn_output = paddle.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)
        attn_output = paddle.matmul(attn_output, out_absorb.transpose([0, 2, 1]))

        if tuple(attn_output.shape) != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.v_head_dim}, but is {tuple(attn_output.shape)}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape([bsz, q_len, self.num_heads * self.v_head_dim])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DeepseekV2FlashAttention2(DeepseekV2Attention):
    """
    DeepseekV2 flash attention module. This module inherits from `DeepseekV2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_unpad_data(self,attention_mask):
        seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
        indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()  # [2, 1, 1323]
        cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0), (1, 0), data_format="NCL")
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # Note: This function was named _upad_input() in torch transformers/modeling_flash_attention_utils.py
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = self._get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # TODO：cuda error
        key_layer = index_first_axis(
            key_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape([batch_size * kv_seq_len, self.num_heads, head_dim]), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(
                batch_size + 1, dtype=paddle.int32
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(paddle.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:

        output_attentions = False

        bsz, q_len, _ = tuple(hidden_states.shape)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape([bsz, q_len, self.num_heads, self.q_head_dim]).transpose(perm=[0, 2, 1, 3])
        q_nope, q_pe = paddle.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1
        )
        k_pe = k_pe.reshape([bsz, q_len, 1, self.qk_rope_head_dim]).transpose(perm=[0, 2, 1, 3])  # b l 1 d

        # b h l (d_q+d_v)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .reshape([bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim])
            .transpose(perm=[0, 2, 1, 3])
        )

        k_nope, value_states = paddle.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)

        kv_seq_len = tuple(value_states.shape)[-2]
        if past_key_value is not None:
            # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            kv_seq_len += past_key_value[0].shape[1]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = paddle.empty(shape=[bsz, self.num_heads, q_len, self.q_head_dim], dtype=k_pe.dtype)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = paddle.empty(shape=[bsz, self.num_heads, q_len, self.q_head_dim], dtype=k_pe.dtype)  # b h l d
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(
                value_states, pad=[0, self.q_head_dim - self.v_head_dim], pad_from_left_axis=False, data_format="NLC"
            )

        query_states = query_states.transpose(perm=[0, 2, 1, 3])  # b l h d
        key_states = key_states.transpose(perm=[0, 2, 1, 3])
        value_states = value_states.transpose(perm=[0, 2, 1, 3])

        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        dropout_rate = self.attention_dropout if self.training else 0.0

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            None, # attention_mask, # TODO:强制设置为 None 可以跑通
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )  # [b, l, head_dim*head]
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape([bsz, q_len, self.num_heads * self.v_head_dim])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`paddle.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`paddle.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`paddle.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`paddle.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
        """
        causal = self.is_causal and query_length != 1

        if _IS_NPU:
            if attention_mask is not None:
                attn_output = paddle.nn.functional.flash_attention_npu(  # TODO: flash_attn_unpadded
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout=dropout,
                    causal=causal,
                    is_varlen=True,
                )
            else:
                dtype = query_states.dtype
                attn_output = paddle.nn.functional.flash_attention_npu(  # TODO: flash_attn_unpadded
                    query_states.astype("bfloat16"),
                    key_states.astype("bfloat16"), 
                    value_states.astype("bfloat16"),
                    attn_mask=attention_mask,
                    dropout=dropout,
                    causal=causal,
                )
                attn_output = attn_output.astype(dtype)
        else:
            head_dim = query_states.shape[-1]
            if softmax_scale is None:
                softmax_scale = head_dim**-0.5

            if attention_mask is not None:  # attention_mask.shape # [2, 1, 1323, 1323]
                batch_size = query_states.shape[0]  # [2, 1323, 12, 128]
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                attn_output_unpad = flash_attn_varlen_func(  # flash_attn_unpadded
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout=dropout, # not dropout_p=
                    scale=softmax_scale,  # not softmax_scale=
                    causal=causal,
                )[0]

                attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    causal=causal,  # no softmax_scale=
                )[0]

        return attn_output


# from paddlenlp.transformers.deepseek_v2.modeling import DeepseekV2Attention # diff
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention": DeepseekV2FlashAttention2,

    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention": DeepseekV2FlashAttention2,

    "mha_eager": LlamaAttention,
    "mha_flash_attention": LlamaAttention,  # 没有LlamaFlashAttention2
}


class DeepseekV2DecoderLayer(paddle.nn.Layer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_mla:
            attn_implementation = "mla_" + config.get("_attn_implementation", "flash_attention")
        else:
            attn_implementation = "mha_" + config.get("_attn_implementation", "flash_attention")
        if config.use_mla:
            self.self_attn = ATTENTION_CLASSES[attn_implementation](config=config, layer_idx=layer_idx)
        else:
            self.self_attn = ATTENTION_CLASSES[attn_implementation](config=config)
        # self.self_attn = DeepseekV2Attention(config=config, layerwise_recompute=layer_idx)

        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(config, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config, config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,  #
            **kwargs,
        )

        if len(attn_output) == 3:
            hidden_states, self_attn_weights, present_key_value = attn_output
        elif len(attn_output) == 2:
            hidden_states, self_attn_weights = attn_output
            present_key_value = None
        else:
            raise ValueError

        # Fully Connected
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class DeepseekV2PreTrainedModel(PretrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"  #
    _no_split_modules = ["DeepseekV2DecoderLayer"]

    def _init_weights(self, module):
        with paddle.no_grad():
            std = self.config.initializer_range
            if isinstance(module, paddle.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, paddle.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module._padding_idx is not None:
                    module.weight.data[module._padding_idx].zero_()


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make casual mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    if get_env_device() == "npu":
        mask = paddle.tril(paddle.ones((target_length, target_length))).astype("int32")
    else:
        mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    if get_env_device() == "npu":
        mask = mask[:, None, None, :].astype(dtype)
    else:
        mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.LayerList(
            [DeepseekV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV2RMSNorm(config, config.hidden_size, eps=config.rms_norm_eps)

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        # self._post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        if get_env_device() == "xpu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(-1.7005809656952787e38, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask, x, y)
        else:
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(
                dtype
            )
        return expanded_attn_mask

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor],
        attention_mask: paddle.Tensor,
        output_attentions: bool,
        past_key_value: paddle.Tensor,
        use_cache: bool,
        # attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            # attn_mask_startend_row_indices,
            use_reentrant=False,  # self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = tuple(input_ids.shape)[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = tuple(inputs_embeds.shape)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        # [bs, seq_len]
        attention_mask = (
            paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            if attention_mask is None
            else attention_mask
        )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), past_key_values_length, inputs_embeds.dtype
        )  # [bs, 1, seq_len, seq_len]
        if self.config.use_flash_attention:
            attention_mask = None if is_casual_mask(attention_mask) else attention_mask

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if self.enable_recompute and has_gradient:
                layer_outputs = self.recompute_training_full(
                    # if self.enable_recompute and self.training:
                    #     layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.CausalLMOutputWithPast]:
        """
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            # cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.cast("float32")
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = paddle.nn.CrossEntropyLoss(reduction="sum")
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            loss = loss_fct(shift_logits, shift_labels)
            label_sum = paddle.sum(shift_labels != -100).cast("float32")
            loss = loss / label_sum

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return paddlenlp.transformers.model_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
            cache_length = past_length = past_key_values[0][0].shape[1]

            max_cache_length = None
            if attention_mask is not None and tuple(attention_mask.shape)[1] > tuple(input_ids.shape)[1]:
                input_ids = input_ids[:, -(tuple(attention_mask.shape)[1] - past_length) :]
            elif past_length < tuple(input_ids.shape)[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + tuple(input_ids.shape)[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype(dtype="int64").cumsum(axis=-1) - 1
            position_ids.masked_fill_(mask=attention_mask == 0, value=1)
            if past_key_values:
                position_ids = position_ids[:, -tuple(input_ids.shape)[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if (
            isinstance(outputs, paddlenlp.transformers.model_outputs.CausalLMOutputWithPast)
            and "past_key_values" in outputs
        ):
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )[:, :, -1:, :]

        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
