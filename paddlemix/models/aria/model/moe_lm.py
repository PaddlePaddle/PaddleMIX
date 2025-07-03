# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
from typing import Tuple

import paddle
from paddlenlp.transformers import LlamaConfig
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.llama.modeling import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

logger = logging.getLogger(__name__)


class AriaMoELMConfig(LlamaConfig):
    """
    Configuration class for AriaMoE language model.

    This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.
    """

    model_type = "aria_moe_lm"

    def __init__(
        self,
        moe_intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_z_loss_coeff: float = 1e-05,
        moe_aux_loss_coeff: float = 0.001,
        moe_num_shared_experts: int = 2,
        **kwargs
    ):
        """
        Initialize the AriaMoELMConfig.

        Args:
            moe_intermediate_size (int): The intermediate size for MoE layers. Default is 4096.
            moe_num_experts (int): The number of experts in the MoE layer. Default is 8.
            moe_topk (int): The number of top experts to route to for each token. Default is 2.
            moe_z_loss_coeff (float): The coefficient for the auxiliary z-loss. Default is 1e-5.
            moe_aux_loss_coeff (float): The coefficient for the auxiliary load balancing loss. Default is 1e-3.
            moe_num_shared_experts (int): The number of shared experts. Default is 2.
            **kwargs: Additional keyword arguments to be passed to the parent LlamaConfig.
        """
        super().__init__(**kwargs)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_num_shared_experts = moe_num_shared_experts

        # new added
        self.mlp_bias = False
        self.hidden_act = "silu"


class MoEAuxLossAutoScaler(paddle.autograd.PyLayer):
    """An AutoScaler that compute and scales the grad for auxiliary loss."""

    main_loss_backward_scale: paddle.Tensor = paddle.to_tensor(data=1.0)

    @staticmethod
    def forward(ctx, output: paddle.Tensor, aux_loss: paddle.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        """Class Attribute: torch.autograd.function.FunctionCtx.saved_tensors, can not convert, please check whether it is torch.Tensor.*/torch.autograd.function.FunctionCtx.*/torch.distributions.Distribution.* and convert manually"""
        (aux_loss,) = ctx.saved_tensors()
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = paddle.ones_like(x=aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: paddle.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """
    z_loss = paddle.mean(x=paddle.square(x=paddle.logsumexp(x=logits, axis=-1))) * z_loss_coeff
    return z_loss


def switch_load_balancing_loss_func(
    probs: paddle.Tensor,
    tokens_per_expert: paddle.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
):
    """Calculate the auxiliary loss for better load balancing.
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): The softmax probs output by the router for each token. [num_tokens, num_experts]
        tokens_per_expert (torch.Tensor): The number of assigned tokens for each expert. [num_experts]

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_tokens = tuple(probs.shape)[0] * topk
    num_experts = tuple(probs.shape)[1]
    probs_mean_per_expert = probs.mean(axis=0)
    tokens_per_expert.to(
        dtype=probs_mean_per_expert.dtype
    )  # Fix the dtype of tokens_per_expert to match the dtype of probs_mean_per_expert
    aux_loss = paddle.sum(x=probs_mean_per_expert * tokens_per_expert) * (
        num_experts / num_tokens * moe_aux_loss_coeff
    )
    return aux_loss


class TopKRouter(paddle.nn.Layer):
    """
    Top-K Router for Mixture of Experts (MoE) models.

    This router determines which experts should process each token based on the top-k scoring experts.
    It also applies auxiliary losses to encourage load balancing among experts.

    Args:
        config (AriaMoELMConfig): Configuration object containing MoE-related parameters.
    """

    def __init__(self, config: AriaMoELMConfig):
        super().__init__()
        self.config = config
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=(self.config.moe_num_experts, self.config.hidden_size))
        )

    def gating(self, input: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the gating logits for each token-expert pair.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size * seq_len, hidden_size].

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size * seq_len, num_experts].
        """
        logits = paddle.nn.functional.linear(x=input, weight=self.weight.T)
        return logits

    def apply_z_loss(self, logits: paddle.Tensor) -> paddle.Tensor:
        """
        Apply z-loss to encourage router logits to remain small for enhanced stability.

        Args:
            logits (torch.Tensor): Router logits.

        Returns:
            torch.Tensor: Logits with z-loss applied.
        """
        z_loss = z_loss_func(logits, self.config.moe_z_loss_coeff)
        logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_aux_loss(
        self,
        logits: paddle.Tensor,
        tokens_per_expert: paddle.Tensor,
        activation: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply auxiliary loss for load balancing among experts.

        Args:
            logits (torch.Tensor): Router logits.
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
            activation (torch.Tensor): Activation values.

        Returns:
            torch.Tensor: Activation with auxiliary loss applied.
        """
        probs = paddle.nn.functional.softmax(x=logits, axis=-1, dtype="float32")
        aux_loss = switch_load_balancing_loss_func(
            probs,
            tokens_per_expert,
            self.config.moe_topk,
            self.config.moe_aux_loss_coeff,
        )
        return MoEAuxLossAutoScaler.apply(activation, aux_loss)

    def routing(self, logits: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Perform the routing operation to determine expert assignments.

        Args:
            logits (torch.Tensor): Router logits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - scores: Softmax probabilities for top-k experts.
                - top_indices: Indices of top-k experts for each token.
                - tokens_per_expert: Number of tokens assigned to each expert.
        """
        if self.training:
            logits = self.apply_z_loss(logits)
        top_logits, top_indices = paddle.topk(k=self.config.moe_topk, x=logits, axis=1)
        scores = paddle.nn.functional.softmax(x=top_logits, axis=-1, dtype="float32").astype(dtype=logits.dtype)

        # Fix 修改这里的histogram计算方式
        flattened_indices = top_indices.flatten()
        tokens_per_expert = paddle.zeros([self.config.moe_num_experts], dtype=flattened_indices.dtype)
        for idx in range(self.config.moe_num_experts):
            tokens_per_expert[idx] = paddle.sum(paddle.cast(flattened_indices == idx, dtype="int64"))

        if self.training:
            scores = self.apply_aux_loss(logits, tokens_per_expert, scores)
        return scores, top_indices, tokens_per_expert

    def forward(self, input: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Forward pass of the TopKRouter.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size * seq_len, hidden_size].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - scores: Softmax probabilities for top-k experts.
                - top_indices: Indices of top-k experts for each token.
                - tokens_per_expert: Number of tokens assigned to each expert.
        """
        logits = self.gating(input)
        logits = logits.reshape(
            shape=(
                -1,
                self.config.moe_num_experts,
            )
        )
        scores, top_indices, tokens_per_expert = self.routing(logits)
        return scores, top_indices, tokens_per_expert


class TokenDispatcher:
    """
    Handles the dispatching and gathering of tokens to and from experts.

    This class is responsible for permuting tokens based on expert assignments and
    unpermuting them after expert processing.

    Args:
        config (AriaMoELMConfig): Configuration object containing MoE-related parameters.
    """

    def __init__(self, config: AriaMoELMConfig):
        self.config = config
        self.hidden_states_shape = None
        self.reversed_input_permutation_mapping = None

    def token_permutation(self, hidden_states: paddle.Tensor, indices: paddle.Tensor) -> paddle.Tensor:
        """
        Permute tokens based on expert assignments.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            indices (torch.Tensor): Expert assignment indices.

        Returns:
            torch.Tensor: Permuted tokens.
        """
        self.hidden_states_shape = tuple(hidden_states.shape)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                hidden_states.shape[-1],
            )
        )
        flatten_indices = indices.flatten()
        sorted_indices = paddle.argsort(x=flatten_indices, stable=True)
        permuted_tokens = hidden_states.index_select(axis=0, index=sorted_indices // self.config.moe_topk)
        self.reversed_input_permutation_mapping = sorted_indices
        return permuted_tokens

    def token_unpermutation(self, permuted_tokens: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        """
        Unpermute tokens and combine expert outputs.

        Args:
            permuted_tokens (torch.Tensor): Tokens after expert processing.
            scores (torch.Tensor): Expert assignment scores.

        Returns:
            torch.Tensor: Unpermuted and combined output.
        """
        num_unpermuted_tokens = scores.size
        unpermuted_tokens = paddle.zeros(
            shape=(num_unpermuted_tokens, permuted_tokens.shape[1]),
            dtype=permuted_tokens.dtype,
        )
        unpermuted_tokens.scatter_(self.reversed_input_permutation_mapping, permuted_tokens)
        unpermuted_tokens = unpermuted_tokens.reshape(shape=(-1, self.config.moe_topk, permuted_tokens.shape[1]))
        unpermuted_tokens = unpermuted_tokens * scores.unsqueeze(axis=-1)
        unpermuted_tokens = unpermuted_tokens.sum(axis=1).astype(dtype=permuted_tokens.dtype)
        output = unpermuted_tokens.reshape(shape=self.hidden_states_shape)
        return output


class SharedExpertMLP(LlamaMLP):
    """
    Shared Expert MLP for shared experts.

    Unlike routed experts, shared experts process all tokens without routing.
    This class reconfigures the intermediate size in comparison to the LlamaMLP.

    Args:
        config (AriaMoELMConfig): Configuration object for the AriaMoE language model.
    """

    def __init__(self, config: AriaMoELMConfig):
        paddle.nn.Layer.__init__(self)  # TODO: 需要检查是否正确
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size * config.moe_num_shared_experts
        self.gate_proj = paddle.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias_attr=config.mlp_bias,
        )
        self.up_proj = paddle.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias_attr=config.mlp_bias,
        )
        self.down_proj = paddle.nn.Linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias_attr=config.mlp_bias,
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self.fuse_attention_ffn = config.fuse_attention_ffn  # Fix


def sequential_gemm(input, weight, tokens_per_expert):
    """
    Compute the matrix multiplication (GEMM) for each expert sequentially. This approach is computationally inefficient, especially when dealing with a large number of experts.

    Args:
        input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
        weight (torch.Tensor): Weight tensor of shape (num_experts, in_features, out_features).
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

    Returns:
        torch.Tensor: Output tensor of shape (num_tokens, out_features).
    """
    num_tokens = tuple(input.shape)[0]
    out_features = tuple(weight.shape)[-1]
    output = paddle.zeros(shape=[num_tokens, out_features], dtype=input.dtype)
    cumsum_num_tokens = paddle.cumsum(x=tokens_per_expert, axis=0)
    zero_tensor = paddle.zeros(shape=[1], dtype=input.dtype)
    cumsum_num_tokens = paddle.concat(x=(zero_tensor, cumsum_num_tokens))
    for expert_num in range(tuple(weight.shape)[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        # Fix Skip if no tokens are assigned to this expert
        if start == end:
            continue
        tokens = input[int(start) : int(end)]
        out = paddle.matmul(x=tokens, y=weight[expert_num])
        output[int(start) : int(end)] = out
    return output


try:
    from grouped_gemm.ops import gmm

    if os.environ.get("USE_GROUPED_GEMM", "1") == "0":
        logger.warning("USE_GROUPED_GEMM is set to 0, using sequential GEMM.")
        experts_gemm = sequential_gemm
    else:
        experts_gemm = gmm
except ImportError:
    logger.warning("`grouped_gemm` is not installed, using sequential GEMM.")
    experts_gemm = sequential_gemm


class GroupedGEMM(paddle.nn.Layer):
    """
    Grouped GEMM (General Matrix Multiplication) module for efficient expert computation.
    This module utilizes the grouped_gemm library (https://github.com/fanshiqing/grouped_gemm)
    for optimized performance. If the grouped_gemm library is not installed, it gracefully
    falls back to a sequential GEMM implementation, which may be slower but ensures
    functionality.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        groups (int): Number of expert groups.
    """

    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[groups, in_features, out_features])
        )

    def forward(self, input, tokens_per_expert):
        """
        Perform grouped matrix multiplication.

        Args:
            input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor of shape (num_tokens, out_features).
        """
        tokens_per_expert = tokens_per_expert.cpu()
        paddle.device.set_device(paddle.device.get_device())
        return experts_gemm(input, self.weight, tokens_per_expert)


class GroupedMLP(paddle.nn.Layer):
    """
    Grouped MLP module for Mixture of Experts.

    Args:
        config (AriaMoELMConfig): Configuration object for the model.
    """

    def __init__(self, config: AriaMoELMConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = GroupedGEMM(config.hidden_size, config.moe_intermediate_size * 2, config.moe_num_experts)
        self.fc2 = GroupedGEMM(config.moe_intermediate_size, config.hidden_size, config.moe_num_experts)

        def glu(x):
            x = paddle.chunk(x=x, chunks=2, axis=-1)
            return paddle.nn.functional.silu(x=x[0]) * x[1]

        self.activation_func = glu

    def forward(self, permuted_tokens, tokens_per_expert):
        """
        Forward pass of the Grouped MLP.

        Args:
            permuted_tokens (torch.Tensor): Permuted input tokens.
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        fc1_output = self.fc1(permuted_tokens, tokens_per_expert)
        fc1_output = self.activation_func(fc1_output)
        fc2_output = self.fc2(fc1_output, tokens_per_expert)
        return fc2_output


class MoELayer(paddle.nn.Layer):
    """
    Mixture of Experts (MoE) Layer for the AriaMoE model.

    This layer implements the MoE mechanism, which routes input tokens to different experts
    based on a routing algorithm, processes them through the experts, and then combines
    the outputs.

    Args:
        config (AriaMoELMConfig): Configuration object for the MoE layer.
    """

    def __init__(self, config: AriaMoELMConfig):
        super().__init__()
        self.router = TopKRouter(config)
        self.token_dispatcher = TokenDispatcher(config)
        self.experts = GroupedMLP(config)
        self.shared_experts = SharedExpertMLP(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass of the MoE Layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.

        Process:
        1. Route tokens to experts using the router.
        2. Permute tokens based on routing decisions.
        3. Process tokens through experts.
        4. Unpermute and combine expert outputs.
        5. Add shared expert output to the final result.
        """
        scores, indices, tokens_per_expert = self.router(hidden_states)
        permuted_tokens = self.token_dispatcher.token_permutation(hidden_states, indices)
        expert_output = self.experts(permuted_tokens, tokens_per_expert)
        output = self.token_dispatcher.token_unpermutation(expert_output, scores)
        shared_expert_output = self.shared_experts(hidden_states)
        output += shared_expert_output
        return output


class MoEDecoderLayer(LlamaDecoderLayer):
    """
    Custom Decoder Layer for the AriaMoE model which modifies the standard `LlamaDecoderLayer` by
    replacing the traditional MLP with a Mixture of Experts (MoE) Layer.

    Args:
        config (LlamaConfig): Configuration object for the layer.
        layer_idx (int): Index of the current layer in the model.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        paddle.nn.Layer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = MoELayer(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
        self.enable_recompute = False


class AriaMoELMModel(LlamaModel):
    """
    Custom LlamaModel for the AriaMoE model which modifies the standard LlamaModel by
    replacing the `LlamaDecoderLayer` with `MoEDecoderLayer`.

    This model implements a Mixture of Experts (MoE) approach, where each layer contains
    multiple expert networks that specialize in different aspects of the input.

    Args:
        config (LlamaConfig): Configuration object for the model.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[MoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config)
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=config.hidden_size, max_position_embeddings=config.max_position_embeddings
        )
        self.gradient_checkpointing = False


class AriaMoELMForCausalLM(LlamaForCausalLM):
    """
    AriaMoE model for causal language modeling tasks.

    This class extends LlamaForCausalLM to incorporate the Mixture of Experts (MoE) approach,
    allowing for more efficient and scalable language modeling.

    Args:
        config (AriaMoELMConfig): Configuration object for the model.
    """

    _tied_weights_keys = ["lm_head.weight"]
    config_class = AriaMoELMConfig
    _no_split_modules = ["MoEDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = AriaMoELMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = paddle.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias_attr=False,
        )

    def set_z_loss_coeff(self, z_loss_coeff: float):
        """
        Set the coefficient for the z-loss in the MoE routing.

        Args:
            z_loss_coeff (float): The coefficient for the z-loss.
        """
        self.config.moe_z_loss_coeff = z_loss_coeff

    def set_aux_loss_coeff(self, aux_loss_coeff: float):
        """
        Set the coefficient for the auxiliary loss in the MoE routing.

        Args:
            aux_loss_coeff (float): The coefficient for the auxiliary loss.
        """
        self.config.moe_aux_loss_coeff = aux_loss_coeff
