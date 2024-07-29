from paddle import nn
import paddle
import paddle.nn.functional as F

class ZKKFacebookDIT(nn.Layer):
    def __init__(self, num_layers: int, dim: int, num_attention_heads: int, attention_head_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.dtype = "float16"
        self.attention_head_dim = attention_head_dim

        self.fcs0 = nn.LayerList([nn.Linear(256, 1152) for i in range(self.num_layers)])
        self.fcs1 = nn.LayerList([nn.Linear(1152, 1152) for i in range(self.num_layers)])
        self.fcs2 = nn.LayerList([nn.Linear(1152, 6912) for i in range(self.num_layers)])
        self.embs = nn.LayerList([nn.Embedding(1001, 1152) for i in range(self.num_layers)])

        self.qkv = nn.LayerList([nn.Linear(dim, dim * 3) for i in range(self.num_layers)])
        self.out_proj = nn.LayerList([nn.Linear(dim, dim) for i in range(self.num_layers)])
        self.ffn1 = nn.LayerList([nn.Linear(dim, dim*4) for i in range(self.num_layers)])
        self.ffn2 = nn.LayerList([nn.Linear(dim*4, dim) for i in range(self.num_layers)])

    @paddle.incubate.jit.inference(enable_new_ir=True, 
                          cache_static_model=True,
                          exp_enable_use_cutlass=True,
                          delete_pass_lists=["add_norm_fuse_pass"],
                        )
    def forward(self,hidden_states, timestep, class_labels):

        tmp = paddle.arange(dtype='float32', end=128)
        tmp = tmp * -9.21034049987793 * 0.007874015718698502
        tmp = paddle.exp(tmp).reshape([1,128])

        timestep = timestep.cast("float32")
        timestep = timestep.reshape([2,1])

        tmp = tmp * timestep

        tmp = paddle.concat([paddle.cos(tmp), paddle.sin(tmp)], axis=-1)
        common_tmp = tmp.cast(self.dtype)

        for i in range(self.num_layers):
            tmp = self.fcs0[i](common_tmp)
            tmp = F.silu(tmp)
            tmp = self.fcs1[i](tmp)
            tmp = tmp + self.embs[i](class_labels)
            tmp = F.silu(tmp)
            tmp = self.fcs2[i](tmp)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = tmp.chunk(6, axis=1)
            norm_hidden_states = paddle.incubate.tt.adaptive_layer_norm(hidden_states, scale_msa, shift_msa)
            q,k,v = self.qkv[i](norm_hidden_states).chunk(3, axis=-1)
            q = q.reshape([2,256,16,72])
            k = k.reshape([2,256,16,72])
            v = v.reshape([2,256,16,72])
            
            norm_hidden_states = F.scaled_dot_product_attention_(q, k, v, scale=self.attention_head_dim**-0.5)
            norm_hidden_states = norm_hidden_states.reshape([2,256,1152])
            norm_hidden_states = self.out_proj[i](norm_hidden_states)

            hidden_states = hidden_states + norm_hidden_states * gate_msa.reshape([2,1,1152])
            
            norm_hidden_states = paddle.incubate.tt.adaptive_layer_norm(hidden_states, scale_mlp, shift_mlp)

            norm_hidden_states = self.ffn1[i](norm_hidden_states)
            norm_hidden_states = F.gelu(norm_hidden_states, approximate=True)
            norm_hidden_states = self.ffn2[i](norm_hidden_states)

            hidden_states = hidden_states + norm_hidden_states * gate_mlp.reshape([2,1,1152])
        
        return hidden_states