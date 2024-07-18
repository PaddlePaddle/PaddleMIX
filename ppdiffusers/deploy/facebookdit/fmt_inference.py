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






import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn.functional.flash_attention import flash_attention, flash_attn_unpadded

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.utils import recompute_use_reentrant, use_old_recompute
from ppdiffusers.models.embeddings import LabelEmbedding
from ppdiffusers.models.modeling_utils import ModelMixin
from paddle.nn.initializer import Constant
from paddle.framework import LayerHelper, in_dynamic_mode
from paddle.incubate.nn.functional import (
    fused_layer_norm,
    fused_rms_norm,
)
from paddle.incubate.nn.functional import fused_linear







def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method="gelu",
    compute_dtype="default",
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dynamic_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["dequant_scales"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


class FusedMultiTransformerLayersConfig:
    def __init__(self,
                embed_dim,
                num_heads,
                dim_feedforward: int=0,
                weight_only_quant_bits=-1,  # -1 means use Half precision.
                dropout_rate=0.0,
                activation="geglu",
                norm_type="layernorm",
                use_neox_rotary_style=False,
                normalize_before=True,

                #time_attr
                time_embed_linear1_weight_attr=None, # first linear merge to one linear
                time_embed_linear1_bias_attr=None,
                time_embed_linear2_weight_attrs=None,
                time_embed_linear2_bias_attrs=None,

                #adaLN_linear
                adaLN_modulate_linear_weight_attrs=None,
                adaLN_modulate_linear_bias_attrs=None,

                #attention_attr
                ln_scale_attrs=None,
                ln_bias_attrs=None,
                qkv_weight_attrs=None,
                qkv_weight_scale_attrs=None,
                qkv_bias_attrs=None,
                q_norm_weight_attrs=None,
                q_norm_bias_attrs=None,
                k_norm_weight_attrs=None,
                k_norm_bias_attrs=None,
                linear_weight_attrs=None,
                linear_weight_scale_attrs=None,
                linear_bias_attrs=None,

                ##ffn_attr
                ffn_ln_scale_attrs=None,
                ffn_ln_bias_attrs=None,
                ffn1_weight_attrs=None,
                ffn1_weight_scale_attrs=None,
                ffn1_bias_attrs=None,
                ffn2_weight_attrs=None,
                ffn2_weight_scale_attrs=None,
                ffn2_bias_attrs=None,
                qkv_out_scale_attrs=None,
                ffn1_out_scale_attrs=None,
                ffn2_out_scale_attrs=None,
                epsilon=1e-5,
                residual_alpha=1.0,
                num_layers=-1,
                nranks=1,
                trans_qkvw=False,
                ring_id=-1,
                skip=False
                ):



        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.norm_type = norm_type


        #TimeEmbed
        self.time_embed_linear1_weight_attr = time_embed_linear1_weight_attr
        self.time_embed_linear1_bias_attr = time_embed_linear1_bias_attr 
        self.time_embed_linear2_weight_attrs = time_embed_linear2_weight_attrs
        self.time_embed_linear2_bias_attrs = time_embed_linear2_bias_attrs

        #adaLN_linear
        self.adaLN_linear_weight_attrs = adaLN_modulate_linear_weight_attrs
        self.adaLN_linear_bias_attrs = adaLN_modulate_linear_bias_attrs



        #Attention
        self.qkv_weight_attrs = qkv_weight_attrs
        self.qkv_weight_scale_attrs = qkv_weight_scale_attrs
        self.qkv_bias_attrs = qkv_bias_attrs

        self.q_norm_weight_attrs=q_norm_weight_attrs
        self.q_norm_bias_attrs=q_norm_bias_attrs

        self.k_norm_weight_attrs=k_norm_weight_attrs
        self.k_norm_bias_attrs=k_norm_bias_attrs


        self.linear_weight_attrs = linear_weight_attrs
        self.linear_weight_scale_attrs = linear_weight_scale_attrs
        self.linear_bias_attrs = linear_bias_attrs

        ### ffn

        self.ffn_ln_scale_attrs = ffn_ln_scale_attrs
        self.ffn_ln_bias_attrs = ffn_ln_bias_attrs
        ##### ffn1
        self.ffn1_weight_attrs = ffn1_weight_attrs
        self.ffn1_weight_scale_attrs = ffn1_weight_scale_attrs
        self.ffn1_bias_attrs = ffn1_bias_attrs

        ##### ffn2
        self.ffn2_weight_attrs = ffn2_weight_attrs
        self.ffn2_weight_scale_attrs = ffn2_weight_scale_attrs
        self.ffn2_bias_attrs = ffn2_bias_attrs

        self.qkv_out_scale_attrs = qkv_out_scale_attrs
        self.ffn1_out_scale_attrs = ffn1_out_scale_attrs
        self.ffn2_out_scale_attrs = ffn2_out_scale_attrs


        self.epsilon = epsilon
        self.residual_alpha = residual_alpha
        self.num_layers = num_layers
        self.nranks = nranks
        self.trans_qkvw = trans_qkvw
        self.ring_id = ring_id
        pass


class FusedMultiTransformerLayers(nn.Layer):
    def __init__(self, 
                config: FusedMultiTransformerLayersConfig, export = False):
        super().__init__()
        self.num_layers = config.num_layers
        assert config.embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(
            config.embed_dim
        )
        assert config.num_heads > 0, "Expected nhead to be greater than 0, " "but received {}".format(config.num_heads)
        assert config.dim_feedforward > 0, "Expected dim_feedforward to be greater than 0, but received {}".format(
            config.dim_feedforward
        )
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.export = export
        self.mlp_hidden_dim = config.dim_feedforward
        self.dim_feedforward = config.dim_feedforward
        self._dtype = self._helper.get_default_dtype()
        self._norm_weight_dtype = 'float32'
        self.create_params_type = self._dtype
        self.activation = config.activation
        self._epsilon = config.epsilon
        self._ring_id = config.ring_id
        self.nranks = config.nranks
        self.norm_type = config.norm_type
        if self.norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        elif self.norm_type == "rmsnorm":
            self.norm_func = fused_rms_norm
        else:
            raise NotImplementedError(f"Only support norm type of [layernorm, rmsnorm], but got {self.norm_type}")
        self.linear = fused_linear


        #time_embed_linear1
        self.time_embed_linear1_weight = self.create_parameter(
            attr=config.time_embed_linear1_weight_attr,
            shape=[256, config.embed_dim * self.num_layers],
            default_initializer=Constant(value=1.0),
            dtype=self.create_params_type,
        )
        time_embed_linear1_bias = None
        if config.time_embed_linear1_bias_attr:
            self.time_embed_linear1_bias = self.create_parameter(
                attr=config.time_embed_linear1_bias_attr,
                shape=[config.embed_dim * self.num_layers],
                is_bias=True,
                dtype=self.create_params_type,
            )
        
        self._add_parameter(self.time_embed_linear1_weight)
        self._add_parameter(self.time_embed_linear1_bias)


        #time_embed_linear2
        self.time_embed_linear2_weights, self.time_embed_linear2_biases = [], []


        #adaLN_linear
        self.adaLN_linear_weights, self.adaLN_linear_biases = [], [] 
       

        #attention
        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []


        #ffn
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []



        

        for i in range(self.num_layers):
            time_embed_linear2_weight_attr = self.get_attr(config.time_embed_linear2_weight_attrs, i)
            time_embed_linear2_bias_attr = self.get_attr(config.time_embed_linear2_bias_attrs, i)
            
            
            adaLN_linear_weight_attr = self.get_attr(config.adaLN_linear_weight_attrs, i)
            adaLN_linear_bias_attr = self.get_attr(config.adaLN_linear_bias_attrs, i)


            
            qkv_weight_attr = self.get_attr(config.qkv_weight_attrs, i)
            qkv_bias_attr = self.get_attr(config.qkv_bias_attrs, i)
            linear_weight_attr = self.get_attr(config.linear_weight_attrs, i)
            linear_bias_attr = self.get_attr(config.linear_bias_attrs, i)


            ffn1_weight_attr = self.get_attr(config.ffn1_weight_attrs, i)
            ffn1_bias_attr = self.get_attr(config.ffn1_bias_attrs, i)
            ffn2_weight_attr = self.get_attr(config.ffn2_weight_attrs, i)
            ffn2_bias_attr = self.get_attr(config.ffn2_bias_attrs, i)


            time_embed_linear2_weight = self.create_parameter(
                shape=[config.embed_dim, config.embed_dim],
                attr=time_embed_linear2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            time_embed_linear2_bias = self.create_parameter(
                shape=[config.embed_dim],
                attr=time_embed_linear2_bias_attr,
                dtype=self.create_params_type,
                is_bias=True,
            )

            self.time_embed_linear2_weights.append(time_embed_linear2_weight)
            self.time_embed_linear2_biases.append(time_embed_linear2_bias)

            self._add_parameter(time_embed_linear2_weight)
            self._add_parameter(time_embed_linear2_bias)


            adaLN_linear_weight = self.create_parameter(
                shape=[config.embed_dim, config.embed_dim * 6],
                attr=adaLN_linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=True,
            )
            adaLN_linear_bias = self.create_parameter(
                shape=[config.embed_dim * 6],
                attr=adaLN_linear_bias_attr,
                dtype=self.create_params_type,
                is_bias=True, 
            )
            self.adaLN_linear_weights.append(adaLN_linear_weight)
            self.adaLN_linear_biases.append(adaLN_linear_bias)

            self._add_parameter(adaLN_linear_weight)
            self._add_parameter(adaLN_linear_bias)


            self.init_weight_shape(config)
            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[3 * self.num_heads * self.head_dim],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            # TODO(wangbojun), ffn shape need refine 
            ffn1_weight = self.create_parameter(
                shape=self.ffn1_weight_shape,
                attr=ffn1_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn1_bias = None
            if ffn1_bias_attr:
                ffn1_bias = self.create_parameter(
                    shape=[int(self.dim_feedforward)],
                    attr=ffn1_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn2_weight = self.create_parameter(
                shape=self.ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn2_bias = None
            if ffn2_bias_attr:
                ffn2_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn2_bias_attr,
                    dtype=self.create_params_type,
                    is_bias=True,
                )



            
            
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)



            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            




            self._add_parameter(qkv_weight)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_weight)
            self._add_parameter(linear_bias)


            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn1_bias)
            self._add_parameter(ffn2_weight)
            self._add_parameter(ffn2_bias)
        pass

    def norm_func_wrap(self, x, norm_weight=None, norm_bias=None, epsilon=1e-8, begin_norm_axis=1, bias=None, residual=None):
        if self.export:
            return self.norm_func(x=x, norm_weight=norm_weight, norm_bias=norm_bias, epsilon=epsilon, begin_norm_axis=begin_norm_axis, bias=bias, residual=residual)
        else:
            return self.norm_func(x=x, norm_weight=norm_weight, norm_bias=norm_bias, epsilon=epsilon, begin_norm_axis=begin_norm_axis, bias=bias, residual=residual)[0]
    
    def get_attr(self, attrs, idx):
        if isinstance(attrs, (list, tuple)):
            assert (
                len(attrs) == self.num_layers
            ), f"length of attrs is {len(attrs)} is not equal to self.num_layers {self.num_layers}"
            return attrs[idx]
        return attrs

    def _add_parameter(self, param):
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param


    def init_weight_shape(self, config):
        self.qkv_weight_shape = (
            [3 * self.num_heads * self.head_dim, self.embed_dim]
            if config.trans_qkvw
            else [self.embed_dim, 3 * self.num_heads * self.head_dim]
        )
        self.linear_weight_shape = [self.num_heads * self.head_dim, self.embed_dim]        
        ffn2_hidden_features = int(self.dim_feedforward)
        
        self.ffn1_weight_shape = [self.embed_dim, ffn2_hidden_features]

        self.ffn2_weight_shape = [ffn2_hidden_features, self.embed_dim]


    def shift_and_scale(self, x ,shift, scale):
        bs, dim = scale.shape
        x = x.reshape([bs,-1,dim])
        out = x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)
        return out.reshape([-1,dim])

    def compute_time_emd_liner1(self, timestep_proj):
        linear_out = fused_linear(timestep_proj, self.time_embed_linear1_weight, self.time_embed_linear1_bias)
        act_out = nn.functional.silu(linear_out)
        return paddle.split(act_out, num_or_sections=self.num_layers, axis=1)

    def compute_time_emd_liner2(self, input, i, label_proj):
        time_embedding = fused_linear(input, self.time_embed_linear2_weights[i], self.time_embed_linear2_biases[i])
        condition_befor_act = time_embedding + label_proj
        condition = nn.functional.silu(condition_befor_act)
        modulate_out = fused_linear(condition, self.adaLN_linear_weights[i], self.adaLN_linear_biases[i])
        return modulate_out, condition_befor_act




    def compute_layernorm_before_qkv(self, src, i):
        ln_out = self.norm_func_wrap(x=src, norm_weight=None, norm_bias=None, epsilon=self._epsilon, begin_norm_axis=1)
        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if float(paddle.version.cuda()) < 11.6:
            qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
            if self.qkv_biases[i] is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
            return qkv_out
        else:
            # This method requires CUDA version >= 11.6.
            return self.linear(ln_out, self.qkv_weights[i], self.qkv_biases[i], transpose_weight=False)
        


    def compute_qkv_with_modulate(self, src, residual_input, shift_msa, scale_msa, i):
        ln_out = self.compute_layernorm_before_qkv(src, i)
        modulate_out = self.shift_and_scale(ln_out, shift_msa, scale_msa)
        qkv_out = self.compute_qkv_linear(modulate_out, i)
        return qkv_out, residual_input
    


    def compute_fmha(
        self,
        q_out,
        k_out,
        v_out,
        cu_seq_lens,
        seq_len,
        max_seq_len_q,
        max_seq_len_kv
    ):
        """
        qkv: bsz, seq_len, 3, numhead, headsize ->
        q_out: bsz, numhead, seq_len, headsize
        kv_out: 2, bsz, numhead, seq_len, headsize
        """
        #TODO wangbojun rope
        qktv_out, _ = flash_attn_unpadded(
            q_out,k_out,v_out,
            cu_seq_lens,
            cu_seq_lens,
            max_seq_len_q,
            max_seq_len_kv,
            1.0/math.sqrt(self.head_dim),
            training=False
        )
        qktv_out_reshape = qktv_out.reshape([0,-1])

        return qktv_out_reshape

    def compute_out_linear(self, fmha_out, i):
        return fused_linear(fmha_out, self.linear_weights[i], self.linear_biases[i])
    
    
    
    def compute_attn(
        self,
        qkv_out,
        cu_seq_lens,
        seq_len,
        max_seq_len_q,
        max_seq_len_kv,
        i
    ):
        # fmha compute
        qkv_out = qkv_out.reshape([0,3,self.num_heads, self.head_dim])
        q = qkv_out[:,0,:,:].reshape([-1, self.num_heads, self.head_dim])
        k = qkv_out[:,1,:,:].reshape([-1, self.num_heads, self.head_dim])
        v = qkv_out[:,2,:,:].reshape([-1, self.num_heads, self.head_dim])
        fmha_out = self.compute_fmha(
            q, k, v,
            cu_seq_lens,
            seq_len,
            max_seq_len_q,
            max_seq_len_kv
        )
        out_linear_out = self.compute_out_linear(fmha_out, i)
        return out_linear_out

    def compute_ffn_layernorm(self, out_linear_out, i):
        norm_out = self.norm_func_wrap(
            out_linear_out,
            norm_weight=None,
            norm_bias=None,
            epsilon=self._epsilon,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            # residual=residual_input,
        )
        return norm_out

    def compute_activation(self, ffn1_out, i):
        #TODO(wangbojun)
        return fused_act_bias_wrapper(ffn1_out, self.ffn1_biases[i], act_method="gelu")


    def compute_ffn1(self, tmp_out, i):
        return paddle.matmul(tmp_out, self.ffn1_weights[i])
    def compute_ffn2(self, ffn1_out, i):
        return fused_linear(ffn1_out, self.ffn2_weights[i], self.ffn2_biases[i])



    

    def forward(self, x, time_step_proj, label_proj,
                cu_seq_lens=None,
                img_seq_lens=None,
                max_seq_len_q=None,
                max_seq_len_kv=None):

        time_embed_linear1_out = self.compute_time_emd_liner1(time_step_proj)

        x=x.reshape([-1,self.embed_dim])
        for i in range(self.num_layers):
            adaln_input, condition = self.compute_time_emd_liner2(time_embed_linear1_out[i], i, label_proj)
            if i == 0:
                condition_for_final_layer = condition
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_input.chunk(6, axis=1)
            
            residual_input = x
            qkv_out, _ = self.compute_qkv_with_modulate(x, residual_input, shift_msa, scale_msa, i)
            out_linear_out = self.compute_attn(
                qkv_out,
                cu_seq_lens,
                img_seq_lens,
                max_seq_len_q,
                max_seq_len_kv,
                i
            )

            bs,dim = gate_msa.shape
            out_linear_out=out_linear_out.reshape([bs,-1,dim])
            gate_msa_out = gate_msa.unsqueeze(1) * out_linear_out
            h = residual_input + gate_msa_out.reshape([-1,dim])
            
            
            ffn_ln_out = self.compute_ffn_layernorm(h, i)
            # ffn1 matmul
            ffn1_out = self.compute_ffn1(self.shift_and_scale(ffn_ln_out, shift_mlp, scale_mlp), i)
            ffn1_out = self.compute_activation(ffn1_out, i)
            # ffn2 matmul
            ffn2_out = self.compute_ffn2(ffn1_out, i)
            # out = h + gate_mlp.unsqueeze(1) * ffn2_out
            out = h.reshape([-1,max_seq_len_q, dim]) + gate_mlp.unsqueeze(1) * ffn2_out.reshape([-1,max_seq_len_q, dim])
            x=out.reshape([-1,self.embed_dim])            
        return x, condition_for_final_layer



