import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flash_attention
from paddle.framework import LayerHelper, in_dynamic_mode

class SimplifiedDiTLLaMA2DModel(nn.Layer):
  def __init__(self, 
               num_layers: int, 
               dim: int, 
               n_heads: int, 
               multiple_of: int,
               mlp_ratio: float,
               norm_eps: float):
    super().__init__()
    self.num_layers = num_layers
    self.dim = dim
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.norm_eps = norm_eps
    
    self.adaLN_modulations = nn.LayerList([nn.Linear(min(dim, 1024), 6 * dim) for i in range(num_layers)])
    
    self.attention_norms = nn.LayerList([nn.LayerNorm(dim, epsilon=norm_eps, bias_attr=False) for i in range(num_layers)])
    
    self.wqs = nn.LayerList([nn.Linear(dim, n_heads * self.head_dim, bias_attr=False) for i in range(num_layers)])
    self.wks = nn.LayerList([nn.Linear(dim, n_heads * self.head_dim, bias_attr=False) for i in range(num_layers)])
    self.wvs = nn.LayerList([nn.Linear(dim, n_heads * self.head_dim, bias_attr=False) for i in range(num_layers)])
    self.wos = nn.LayerList([nn.Linear(n_heads * self.head_dim, dim, bias_attr=False) for i in range(num_layers)])
    
    self.q_norms = nn.LayerList([nn.LayerNorm(n_heads * self.head_dim) for i in range(num_layers)])
    self.k_norms = nn.LayerList([nn.LayerNorm(n_heads * self.head_dim) for i in range(num_layers)])
    
    self.ffn_norms = nn.LayerList([nn.LayerNorm(dim, epsilon=norm_eps, bias_attr=False) for i in range(num_layers)])
    
    hidden_dim = int(dim * mlp_ratio)
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of))
    self.w13s = nn.LayerList([nn.Linear(dim, hidden_dim * 2, bias_attr=False) for i in range(num_layers)])
    self.w2s = nn.LayerList([nn.Linear(hidden_dim, dim, bias_attr=False) for i in range(num_layers)])

  def compute_activation(self, 
                           ffn1_out, 
                           bias=None,
                           dequant_scales=None,
                           shift=None,
                           smooth=None,
                           act_method="swiglu",
                           compute_dtype="default",
                           quant_scale=-1,
                           quant_round_type=0,
                           quant_max_bound=0,
                           quant_min_bound=0):
        if in_dynamic_mode():
            out = paddle._C_ops.fused_bias_act(
                ffn1_out,
                bias,
                dequant_scales,
                shift,
                smooth,
                act_method,
                compute_dtype,
                quant_scale,
                quant_round_type,
                quant_max_bound,
                quant_min_bound
            )
            return out
        
        helper = LayerHelper("fused_bias_act")
        out = helper.create_variable_for_type_inference(dtype=ffn1_out.dtype)
        inputs = {}
        inputs["x"] = ffn1_out
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

  @paddle.incubate.jit.inference(cache_static_model=False,
                                 enable_new_ir=True,
                                 exp_enable_use_cutlass=True,)
  def forward(self, x, freqs_cis, adaln_input):
    freqs_cis = paddle.expand(freqs_cis, [-1, self.n_heads, -1, -1])
    adaln_input = F.silu(adaln_input)
    for i in range(self.num_layers):
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulations[i](adaln_input).chunk(6, axis=1)
      ## AdaLN
      import paddlemix
      attn_in = paddlemix.triton_ops.adaptive_layer_norm(x, scale_msa, shift_msa, 
                                                weight=self.attention_norms[i].weight, epsilon=self.norm_eps)
      ## Attention
      xq, xk, xv = self.wqs[i](attn_in), self.wks[i](attn_in), self.wvs[i](attn_in)
      qkv_out = paddle.concat([xq, xk, xv], axis=-1)
      xq, xk, xv = paddlemix.triton_ops.fused_rotary_emb(qkv_out, self.q_norms[i].weight, self.q_norms[i].bias, 
                                              self.k_norms[i].weight, self.k_norms[i].bias, freqs_cis)
      attn_out, _ = flash_attention(xq, xk, xv, dropout=0.0, causal=False, return_softmax=False)
      attn_out = attn_out.flatten(start_axis=-2)
      attn_out = self.wos[i](attn_out)
      ## Fused_adaLN
      resi_out, adaLN_out = paddlemix.triton_ops.fused_adaLN_scale_residual(x, attn_out, gate_msa, scale_mlp, shift_mlp,
                                                          weight=self.ffn_norms[i].weight, epsilon=self.norm_eps)
      ## FFN
      ffn_out = self.w13s[i](adaLN_out)
      ffn_out = self.compute_activation(ffn_out)
      ffn_out = self.w2s[i](ffn_out)
      x = resi_out + gate_mlp.unsqueeze(1) * ffn_out
    return x
    