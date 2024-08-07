from paddle import nn
import paddle
import paddle.nn.functional as F
import math

class SimplifiedFacebookDIT(nn.Layer):
    def __init__(self, num_layers: int, dim: int, num_attention_heads: int, attention_head_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.heads_num = num_attention_heads
        self.head_dim = attention_head_dim
        self.timestep_embedder_in_channels = 256
        self.timestep_embedder_time_embed_dim = 1152
        self.timestep_embedder_time_embed_dim_out = self.timestep_embedder_time_embed_dim
        self.LabelEmbedding_num_classes = 1001
        self.LabelEmbedding_num_hidden_size = 1152
        
        self.fcs0 = nn.LayerList([nn.Linear(self.timestep_embedder_in_channels, 
                                            self.timestep_embedder_time_embed_dim) for i in range(num_layers)])
        
        self.fcs1 = nn.LayerList([nn.Linear(self.timestep_embedder_time_embed_dim,
                                            self.timestep_embedder_time_embed_dim_out) for i in range(num_layers)])
        
        self.fcs2 = nn.LayerList([nn.Linear(self.timestep_embedder_time_embed_dim,
                                            6 * self.timestep_embedder_time_embed_dim) for i in range(num_layers)])
        
        self.embs = nn.LayerList([nn.Embedding(self.LabelEmbedding_num_classes, 
                                               self.LabelEmbedding_num_hidden_size) for i in range(num_layers)])
        

        self.q = nn.LayerList([nn.Linear(dim, dim ) for i in range(num_layers)])
        self.k = nn.LayerList([nn.Linear(dim, dim ) for i in range(num_layers)])
        self.v = nn.LayerList([nn.Linear(dim, dim ) for i in range(num_layers)])
        self.out_proj = nn.LayerList([nn.Linear(dim, dim) for i in range(num_layers)])
        self.ffn1 = nn.LayerList([nn.Linear(dim, dim*4) for i in range(num_layers)])
        self.ffn2 = nn.LayerList([nn.Linear(dim*4, dim) for i in range(num_layers)])

    def forward(self, hidden_states, timesteps, class_labels):
        
        # below code are copied from PaddleMIX/ppdiffusers/ppdiffusers/models/embeddings.py
        num_channels = 256
        max_period = 10000
        downscale_freq_shift = 1
        half_dim = num_channels // 2
        exponent = -math.log(max_period) * paddle.arange(start=0, end=half_dim, dtype="float32")
        exponent = exponent / (half_dim - downscale_freq_shift)
        emb = paddle.exp(exponent)
        emb = timesteps[:, None].cast("float32") * emb[None, :]
        emb = paddle.concat([paddle.cos(emb), paddle.sin(emb)], axis=-1)
        common_emb = emb.cast(hidden_states.dtype)
        
        for i in range(self.num_layers):  
            emb = self.fcs0[i](common_emb)
            emb = F.silu(emb)
            emb = self.fcs1[i](emb)
            emb = emb + self.embs[i](class_labels)
            emb = F.silu(emb)
            emb = self.fcs2[i](emb)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)
            import paddlemix
            norm_hidden_states =paddlemix.triton_ops.adaptive_layer_norm(hidden_states, scale_msa, shift_msa)
            q = self.q[i](norm_hidden_states).reshape([0,0,self.heads_num,self.head_dim])
            k = self.k[i](norm_hidden_states).reshape([0,0,self.heads_num,self.head_dim])
            v = self.v[i](norm_hidden_states).reshape([0,0,self.heads_num,self.head_dim])

            norm_hidden_states = F.scaled_dot_product_attention_(q, k, v, scale=self.head_dim**-0.5)
            norm_hidden_states = norm_hidden_states.reshape([norm_hidden_states.shape[0],norm_hidden_states.shape[1],self.dim])
            norm_hidden_states = self.out_proj[i](norm_hidden_states)
            
            # hidden_states = hidden_states + norm_hidden_states * gate_msa.reshape([b,1,self.dim])            
            # norm_hidden_states =paddlemix.triton_ops.adaptive_layer_norm(hidden_states, scale_mlp, shift_mlp)
            hidden_states,norm_hidden_states =paddlemix.triton_ops.fused_adaLN_scale_residual(hidden_states, norm_hidden_states, gate_msa, scale_mlp, shift_mlp)

            norm_hidden_states = self.ffn1[i](norm_hidden_states)
            norm_hidden_states = F.gelu(norm_hidden_states, approximate=True)
            norm_hidden_states = self.ffn2[i](norm_hidden_states)

            hidden_states = hidden_states + norm_hidden_states * gate_mlp.reshape([norm_hidden_states.shape[0],1,self.dim])
            
        return hidden_states
    