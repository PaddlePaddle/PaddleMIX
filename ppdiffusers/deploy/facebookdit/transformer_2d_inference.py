# Copyright 2024 The HuggingFace Team. All rights reserved.
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



import json
import math

from fmt_inference import FusedMultiTransformerLayersConfig, FusedMultiTransformerLayers


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_utils import ModelMixin
from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.models.embeddings import PatchEmbed, LabelEmbedding
from ppdiffusers.models.transformer_2d import Transformer2DModelOutput





def get_timestep_embedding(
    timesteps: paddle.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * paddle.arange(start=0, end=half_dim, dtype="float32")

    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = paddle.exp(exponent)
    emb = timesteps[:, None].cast("float32") * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = paddle.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = paddle.concat(emb, paddle.zeros([emb.shape[0], 1]), axis=-1)
    return emb




class DitTransformer2DInferenceModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int = 32,  # image_size // 8
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 8,
        num_layers: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 96,
        mlp_ratio: float = 4.0,
        n_kv_heads=None,
        activation_fn="gelu-approximate",
        multiple_of: int = 256,
        ffn_dim_multiplier=None,
        norm_eps: float = 1e-05,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        qk_norm: bool = True,
        export: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        dim = attention_head_dim * num_attention_heads
        self.emb_dim = dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.dim_ffn = mlp_ratio * dim
        self.norm_eps = norm_eps
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.qk_norm = qk_norm
        self.activation_fn = activation_fn
        self.gradient_checkpointing = True
        self.fused_attn = True
        interpolation_scale = sample_size // 64  # => 64 (= 512 pixart) has interpolation scale 1
        interpolation_scale = max(interpolation_scale, 1)
        self.export = export
        self._dtype = self._helper.get_default_dtype()
        self.x_embedder = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            interpolation_scale=interpolation_scale,

        )
        self.label_embedder = LabelEmbedding(num_classes, dim, class_dropout_prob)



        #finaly_layer
        self.final_layer_norm = nn.LayerNorm(self.emb_dim)
        self.final_layer_linear1 = nn.Linear(in_features = self.emb_dim, out_features = self.emb_dim * 2)
        self.final_layer_linear2 = nn.Linear(in_features = self.emb_dim, out_features = patch_size * patch_size * self.out_channels)

        # time_emd_weight
        fmt_time_emd_linear1_weight_attr = paddle.ParamAttr(
            name="fmt_blocks.time_emd_linear1.weight", initializer=paddle.nn.initializer.Constant(value=0)
        )
        fmt_time_emd_linear1_bias_attr = paddle.ParamAttr(
            name="fmt_blocks.time_emd_linear1.bias", initializer=paddle.nn.initializer.Constant(value=0)
        )
        fmt_time_emd_linear2_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.time_emd_linear2.weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_time_emd_linear2_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.time_emd_linear2.bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        
        
        # adaLN_modulate_linear
        fmt_adaLN_modulate_linear_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks{}.adaLN_modulate_linear.weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        fmt_adaLN_modulate_linear_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks{}.adaLN_modulate_linear.bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        #qkv_weight
        fmt_blocks_attn_qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_attn_qkv_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_qkv_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]


        #out_proj
        fmt_blocks_attn_out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_attn_out_proj_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.out_proj_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        
        #ff1
        fmt_blocks_ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        fmt_blocks_ffn1_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.ffn1_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        #ff2
        fmt_blocks_ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_ffn2_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.ffn2_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]


        self.fmt_config = FusedMultiTransformerLayersConfig(
            embed_dim = self.emb_dim,
            num_heads = self.num_attention_heads,
            dim_feedforward = self.dim_ffn,
            weight_only_quant_bits=-1, #todo(wangbojun)
            dropout_rate=0.0,
            activation="geglu",
            norm_type="layernorm",
            use_neox_rotary_style=False,
            normalize_before=True,
            
            #time_proj
            time_embed_linear1_weight_attr=fmt_time_emd_linear1_weight_attr,
            time_embed_linear1_bias_attr=fmt_time_emd_linear1_bias_attr,
            time_embed_linear2_weight_attrs=fmt_time_emd_linear2_weight_attrs,
            time_embed_linear2_bias_attrs=fmt_time_emd_linear2_bias_attrs,

            #adaLN_modulate_linear
            adaLN_modulate_linear_weight_attrs=fmt_adaLN_modulate_linear_weight_attrs,
            adaLN_modulate_linear_bias_attrs=fmt_adaLN_modulate_linear_bias_attrs,


            #attention all normalizations have no scale and bias.
            qkv_weight_attrs=fmt_blocks_attn_qkv_weight_attrs,
            qkv_bias_attrs = fmt_blocks_attn_qkv_bias_attrs,
            linear_weight_attrs=fmt_blocks_attn_out_proj_weight_attrs,
            linear_bias_attrs = fmt_blocks_attn_out_proj_bias_attrs,

            #ffn
            ffn1_weight_attrs=fmt_blocks_ffn1_weight_attrs,
            ffn1_bias_attrs=fmt_blocks_ffn1_bias_attrs,
            ffn2_weight_attrs=fmt_blocks_ffn2_weight_attrs,
            ffn2_bias_attrs=fmt_blocks_ffn2_bias_attrs,
            
            epsilon=1e-5,
            residual_alpha=1.0,
            num_layers=self.num_layers,
            nranks=1,
            trans_qkvw=False,
            ring_id=-1,
            skip=False
        )
        self.fmt_layer = FusedMultiTransformerLayers(
            self.fmt_config,
            export=self.export,
        )
    def __set_value(self, weight,state_dict, params_name):
        print(f"process weight: {params_name}, param shape is {state_dict[params_name].shape} , var shape is: {weight.shape}, var name : {weight.name}, dtype:{weight.dtype}")
        assert(weight.shape == state_dict[params_name].shape)
        weight.set_value(paddle.to_tensor(state_dict[params_name], dtype=weight.dtype))
        print(f"process weight done")
    

    def set_state_dict(self, state_dict):
        self.__set_value(self.x_embedder.proj.weight, state_dict, "pos_embed.proj.weight")
        self.__set_value(self.x_embedder.proj.bias, state_dict, "pos_embed.proj.bias")

        self.__set_value(self.label_embedder.embedding_table.weight, state_dict, "transformer_blocks.0.norm1.emb.class_embedder.embedding_table.weight")

        transformer_blocks = "transformer_blocks"
        state_dict["time_emb_linear1_weight"] = None
        for i in range(self.fmt_layer.num_layers):

            #time_emdding
            if i==0:
                state_dict["time_emb_linear1_weight"] = state_dict[f"{transformer_blocks}.{i}.norm1.emb.timestep_embedder.linear_1.weight"]
                state_dict["time_emb_linear1_bias"] = state_dict[f"{transformer_blocks}.{i}.norm1.emb.timestep_embedder.linear_1.bias"]
            else:
                state_dict["time_emb_linear1_weight"] = paddle.concat([state_dict["time_emb_linear1_weight"], state_dict[f"{transformer_blocks}.{i}.norm1.emb.timestep_embedder.linear_1.weight"]], axis=-1)
                state_dict["time_emb_linear1_bias"] = paddle.concat([state_dict["time_emb_linear1_bias"], state_dict[f"{transformer_blocks}.{i}.norm1.emb.timestep_embedder.linear_1.bias"]], axis=-1)

            
            self.__set_value(self.fmt_layer.time_embed_linear2_weights[i], state_dict, f"{transformer_blocks}.{i}.norm1.emb.timestep_embedder.linear_2.weight")
            self.__set_value(self.fmt_layer.time_embed_linear2_biases[i], state_dict, f"{transformer_blocks}.{i}.norm1.emb.timestep_embedder.linear_2.bias")

            #shift_and_scale
            self.__set_value(self.fmt_layer.adaLN_linear_weights[i], state_dict, f"{transformer_blocks}.{i}.norm1.linear.weight")
            self.__set_value(self.fmt_layer.adaLN_linear_biases[i], state_dict, f"{transformer_blocks}.{i}.norm1.linear.bias")



            state_dict[f'{transformer_blocks}.{i}.attention.wq.weight'] = state_dict[f'{transformer_blocks}.{i}.attn1.to_q.weight']
            state_dict[f'{transformer_blocks}.{i}.attention.wq.bias'] = state_dict[f'{transformer_blocks}.{i}.attn1.to_q.bias']
            state_dict[f'{transformer_blocks}.{i}.attention.wk.weight'] = state_dict[f'{transformer_blocks}.{i}.attn1.to_k.weight']
            state_dict[f'{transformer_blocks}.{i}.attention.wk.bias'] = state_dict[f'{transformer_blocks}.{i}.attn1.to_k.bias']
            state_dict[f'{transformer_blocks}.{i}.attention.wv.weight'] = state_dict[f'{transformer_blocks}.{i}.attn1.to_v.weight']
            state_dict[f'{transformer_blocks}.{i}.attention.wv.bias'] = state_dict[f'{transformer_blocks}.{i}.attn1.to_v.bias']

            state_dict[f'{transformer_blocks}.{i}.attn.qkv.weight'] = paddle.concat([state_dict[f'{transformer_blocks}.{i}.attention.wq.weight'],
                                                                       state_dict[f'{transformer_blocks}.{i}.attention.wk.weight'],
                                                                       state_dict[f'{transformer_blocks}.{i}.attention.wv.weight']],axis=1)


            state_dict[f'{transformer_blocks}.{i}.attn.qkv.bias'] = paddle.concat([state_dict[f'{transformer_blocks}.{i}.attn1.to_q.bias'],
                                                                       state_dict[f'{transformer_blocks}.{i}.attn1.to_k.bias'],
                                                                       state_dict[f'{transformer_blocks}.{i}.attn1.to_v.bias']])
            self.__set_value(self.fmt_layer.qkv_weights[i], state_dict, f'{transformer_blocks}.{i}.attn.qkv.weight')
            self.__set_value(self.fmt_layer.qkv_biases[i], state_dict, f'{transformer_blocks}.{i}.attn.qkv.bias')
            
            self.__set_value(self.fmt_layer.linear_weights[i],state_dict, f'{transformer_blocks}.{i}.attn1.to_out.0.weight')
            self.__set_value(self.fmt_layer.linear_biases[i],state_dict, f'{transformer_blocks}.{i}.attn1.to_out.0.bias')
            
            
            self.__set_value(self.fmt_layer.ffn1_weights[i],state_dict, f'{transformer_blocks}.{i}.ff.net.0.proj.weight')
            self.__set_value(self.fmt_layer.ffn1_biases[i],state_dict, f'{transformer_blocks}.{i}.ff.net.0.proj.bias')
            
            self.__set_value(self.fmt_layer.ffn2_weights[i],state_dict, f'{transformer_blocks}.{i}.ff.net.2.weight')
            self.__set_value(self.fmt_layer.ffn2_biases[i],state_dict, f'{transformer_blocks}.{i}.ff.net.2.bias')
        self.__set_value(self.fmt_layer.time_embed_linear1_weight, state_dict, "time_emb_linear1_weight")
        self.__set_value(self.fmt_layer.time_embed_linear1_bias, state_dict, "time_emb_linear1_bias")

        self.__set_value(self.final_layer_linear1.weight, state_dict, "proj_out_1.weight")
        self.__set_value(self.final_layer_linear1.bias, state_dict, "proj_out_1.bias")
        self.__set_value(self.final_layer_linear2.weight, state_dict, "proj_out_2.weight")
        self.__set_value(self.final_layer_linear2.bias, state_dict, "proj_out_2.bias")


    def compute_final_layer_and_unpathfiy(self, hidden_states, condition):

        shift, scale = self.final_layer_linear1(F.silu(condition)).chunk(2, axis=1)
        hidden_states = self.final_layer_norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.final_layer_linear2(hidden_states)
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = paddle.transpose(hidden_states, perm=[0, 5, 1, 3, 2, 4])
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )
        return output


    def forward(
        self,
        hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        class_labels: paddle.Tensor,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states: (N, C, H, W) tensor of spatial inputs (images or latent
                representations of images)
            timestep: (N,) tensor of diffusion timesteps
            class_labels: (N,) tensor of class labels
        """
        
        
        hidden_states = hidden_states.cast(self.dtype)
        x = self.x_embedder(hidden_states)

        timestep_proj = get_timestep_embedding(timestep, 256, True, 1).cast(self._dtype)


        label_proj = self.label_embedder(class_labels).cast(self._dtype)

        x_bs, x_seq_len, x_dim = x.shape
        x_seq_lens_tensor =  paddle.full(shape=[x_bs], fill_value=x_seq_len, dtype='int32')
        x_cu_seq_lens_tensor = paddle.concat([paddle.to_tensor([0],dtype='int32'), paddle.cumsum(x_seq_lens_tensor)])
        x_max_seq_lens = x_seq_len

        x, final_condition = self.fmt_layer(x, timestep_proj, label_proj,
                            x_cu_seq_lens_tensor,
                            x_seq_lens_tensor,
                            x_max_seq_lens,
                            x_max_seq_lens
                            )

        return self.compute_final_layer_and_unpathfiy(x.reshape([x_bs, x_seq_len, x_dim]), final_condition)




def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data




# if __name__ == "__main__":
#     paddle.set_device("gpu:7")
#     paddle.set_default_dtype("float16")
#     transformer_path = "/work/caizejun/dit/DiT-XL-2-256/transformer"
#     inference_model = DitTransformer2DInferenceModel(**read_json(transformer_path + "/config.json"))
#     state_dict = paddle.load(transformer_path + "/model_state.pdparams")
#     inference_model.set_state_dict(state_dict)
#     time_step = paddle.load("/work/caizejun/dit/dit_transformer2D_models/inputs/timestep.pdparams")
#     hidden_states = paddle.load("/work/caizejun/dit/params/hidden_states_2D.pdparams")
#     class_labels = paddle.load("/work/caizejun/dit/dit_transformer2D_models/inputs/class_labels.pdparams")
#     res = inference_model(hidden_states, time_step, class_labels)
#     print(res)

    