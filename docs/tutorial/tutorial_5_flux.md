# 前沿多模态模型开发与应用实战第五期：FLUX文生图大模型算法解析与功能体验

多模态生成大模型是近年来生成式人工智能领域的研究热点之一，尤其在文本到图像生成任务中展现出强大的表现力。这类模型通常融合语言和视觉两个模态，通过大规模Transformer架构与扩散式生成机制，实现对文本语义的理解与视觉内容的高质量生成。相比传统的图像生成模型，现代多模态生成模型具备更高的图文一致性、更强的风格控制能力以及更灵活的应用范围，广泛应用于AI绘画、智能设计、游戏制作、广告创意等场景。

本文将聚焦解析开源多模态扩散模型 FLUX，这是一款参数规模高达百亿级的文本到图像生成模型，其在生成质量、图文一致性和效率方面均达到当前领先水平。FLUX 采用了大规模双流 Transformer 架构，结合 Flow Matching 推理策略，能够在较少扩散步数内生成高保真图像。本文将以算法原理与代码实现解析为主线，并结合 PaddleMIX 套件进行实操体验。

## 一、背景
FLUX 模型是由 Black Forest Labs 团队推出的一系列文本生成图像模型，其核心采用了扩散 Transformer架构 。例如，FLUX.1 [dev] 版本包含了约 120 亿参数，是一种校正流（Rectified Flow）Transformer 模型，可以根据文本描述生成高保真图像 。

<div align="center">
    <img src="https://github.com/user-attachments/assets/1551c868-af9c-40f6-a6c6-7e2600072013" alt="Image 2" style="width: 60%;">
    <p style="color: #808080;"> FLUX生成效果展示 </p>
</div>

FLUX 模型构建在扩散模型理论基础之上。扩散模型通过对图像逐步添加噪声并训练模型学习如何去除噪声来生成新图像。在训练中，模型接收加入一定噪声的图像，并预测噪声的成分，使得去噪后能还原原始图像 。这一过程可视为从纯噪声（如高斯噪声）逐步扩散到数据分布（真实图像）的反向过程。然而传统扩散模型的生成轨迹并非严格线性，往往需要数十步迭代采样才能得到清晰图像。

为提升生成效率，FLUX 引入了Rectified Flow技术。这一技术旨在拉直数据到噪声的生成路径，使扩散过程尽可能接近直线 。简单来说，Rectified Flow 将从噪声生成数据的流优化为最短路径，从而大幅减少生成所需步骤 。

需要注意的是，FLUX 并非单一模型，而是一个模型家族，包含多个变体以平衡生成速度与效果，常见的版本包括：
* FLUX.1 [dev]：使用指导蒸馏（Guidance Distillation）技术训练的模型，约120亿参数，生成质量仅次于专业版 。其推理一般需要 ~50 步采样，可支持长文本提示 。
* FLUX.1 [schnell]：使用时间步蒸馏（Timestep Distillation）技术优化的快速模型。“Schnell”在德语中意为快速。该模型要求 guidance_scale=0（不使用额外引导）并限制文本长度不超过 256 个标记 。由于经过多步合并蒸馏，它能在 4 步左右的极少采样下生成合理图像 。
* FLUX.1 [pro]：Black Forest Labs 内部高阶模型，参数规模和生成质量进一步提升，是 FLUX 家族中的顶尖模型（目前未公开权重）。

## 二、模型架构

FLUX 采用纯 Transformer 架构的扩散模型（Diffusion Transformer），这种架构也称为“Rectified Flow Transformer”，旨在结合 Transformer 全局建模能力和扩散模型的逐步生成过程。与 Stable Diffusion v1.x ~ 2.x 相比，FLUX 在架构上有几大显著创新：
* 多模态序列建模： 模型同时处理文本和图像两种模态的序列信息。具体而言，FLUX 使用两个文本编码器将输入提示映射为文本特征序列，同时将图像的潜码（latent）表示展开为图像特征序列。与 SD3 等模型类似，FLUX 对图像潜空间特征采用了2×2 Patch 划分：将潜在特征图按 2×2 区块分组，展平后作为一系列图像token 。这样做可以将原本二维的64通道、H×W大小的latent张量，变为长度为 (H/2)*(W/2) 的序列，每个序列元素维度为 patch_size*patch_size*latent_channels。随后，模型通过线性层将图像patch序列和文本序列投影到统一的特征维度，并拼接在一起，送入 Transformer 进行建模 。该设计使得 Transformer 的自注意力机制能够同时关注文本和图像token，实现跨模态的深度交互。
* 双流注意力机制： 在具体实现上，FLUX Transformer采用了一种双流（Dual-Stream）与单流结合的模块设计。前若干层 Transformer Block 执行双流注意力，即图像序列和文本序列各自经过自注意力和前馈网络，同时通过交叉注意力交互信息。这类似于将文本作为条件，通过交叉注意力影响图像特征，但特别之处在于文本序列本身也被图像特征反向更新。换言之，在双流模块中，图像token和文本token彼此双向注意力，互相影响对方的隐藏状态。这使文本特征能够动态融合图像上下文，而非始终保持静态。经过若干层双流交互后，Transformer 后续层切换为单流模式，主要针对图像序列进行自注意力和特征变换，以细化图像表示 。这种分段式的Transformer结构让模型既能充分利用文本条件指导，又能在后期聚焦图像细节重建。
* 大规模多头注意力： FLUX Transformer 的隐藏维度和多头注意力规模也远超传统扩散模型。其注意力头数为 24，每头维度 128，总的内部特征维度达到 3072（24×128） 。文本和图像特征都被投影到这个3072维空间中进行融合。相比之下，Stable Diffusion v1的UNet隐藏层约为 320~1280 维，注意力头数 8。更高的维度赋予 FLUX 更强的建模能力，但也增加显存需求。为缓解训练/推理压力，FLUX 在注意力模块中引入了门控（Gating）机制：对每一层的注意力输出和MLP输出，均乘以一个可学习的门控参数后再与残差相加 。这种 gating 技术有助于稳定超深Transformer的训练，控制不同层信息流的强度。此外，FLUX 采用了旋转位置嵌入（RoPE）来为图像patch序列提供位置信息，确保 Transformer 知晓每个token对应的空间位置。模型使用三个轴的 RoPE 编码，涵盖二维空间尺寸和patch内局部坐标，以适配高分辨率生成。
* 双文本编码器： 为了更好地理解和表示文本提示，FLUX 引入了双文本编码器架构。它同时使用了 CLIP 和 T5 两种预训练文本模型来编码提示信息 。第一编码器为 CLIP 文本编码器（如 OpenAI CLIP ViT-L/14），长于捕获与视觉相关的语义和风格信息；第二编码器为 T5 编码器（如 T5-v1.1-XXL），擅长理解长文本和复杂描述。FLUX 对这两种编码器的输出加以区分利用：CLIP 文本模型输出的池化文本向量（文本语义的全局Embedding）经线性层投影后，将融合进扩散 Transformer的时间步嵌入，用于指导全局图像风格和语义 ；T5 编码器输出的文本序列特征则经过线性变换后，作为扩散 Transformer交叉注意力的文本token序列。这种双模文本嵌入方法类似于 Stable Diffusion XL 的做法，将语言模型和对比学习模型各自的优势结合，使 FLUX 对文本的理解更加全面。尤其在较长或复杂提示下，T5 编码器允许模型处理多达 512 个标记的文本长度 ，显著超过以往基于 CLIP 的77标记限制。同时，CLIP 提供的全局嵌入可作为一种额外条件，帮助模型更好地对齐视觉语义。例如，FLUX Pipeline 默认会获取 CLIP 文本模型的[EOS]输出作为pooled embedding，并结合时间步嵌入形成扩散模型的条件向量，使模型对提示的整体语义有敏锐感知。
* 扩散指导蒸馏： 在训练技术上，FLUX 引入了 Guidance Distillation（指导蒸馏）和 Timestep Distillation（时间步蒸馏）等新颖策略，以降低推理成本并保持图像质量。其中，指导蒸馏指的是将传统扩散模型依赖的分类器自由引导（CFG）过程融入到单个模型中。通常，扩散模型在推理时需要进行两次前向传递（一次带文本条件，一次不带条件），然后根据 guidance_scale 差异放大关键信号 。通过蒸馏，让模型在训练中直接学习有引导情况下应输出的结果，从而在推理时仅需一次前向传递即可获得接近有引导的效果。简言之，模型本身被训练得更“听话”，减少了对额外引导计算的依赖。这使得 FLUX.1 [dev] 模型在推理时，将 guidance_scale 设为 3.5 这样的中等值即可生成高细节图像 ，同时节省一半计算。时间步蒸馏则进一步针对采样步数进行压缩。通过逐步蒸馏或对抗训练等手段，FLUX.1 [schnell] 模型成功在 4 步迭代内生成清晰图像 （相比原版需要数十步）。 Adversarial Diffusion Distillation 方法即属于此范畴，能将大模型的扩散过程压缩到1~4步内完成，同时输出质量与多步采样相当 。

综上，FLUX 通过Transformer取代U-Net、融合双文本编码、引入校正流损失和蒸馏训练，达到了生成质量和效率上的新高度。在保持高分辨率细节和复杂场景理解方面，FLUX 相比早期稳定扩散模型有显著提升 。

## 三、代码解读
下面以 PaddleMIX 中 FLUX 的实现为例，对关键创新点的代码实现进行讲解。

### FluxTransformer2DModel模块
Flux 的核心模型是 FluxTransformer2DModel，本质上是一个扩散 Transformer。它接收编码后的图像潜空间（通常64通道的latent特征）以及文本嵌入，输出去噪后的图像latent。FluxTransformer2DModel内部堆叠了多层 Transformer 块，其中前几层是双流 Dual-Stream块（同时更新图像和文本流，类似论文中的 MMDiT），后续层是单流 Single-Stream块（仅更新图像流，类似原始 DiT）。以下展示了模型初始化主要组件：
```python
class FluxTransformer2DModel(nn.Layer):
    def __init__(self, ..., num_layers=19, num_single_layers=38, ...):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=768
        )
        # 文本上下文嵌入线性投射：将文本序列embedding降维到inner_dim
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        # 图像latent嵌入线性层：将输入图像通道数投射到inner_dim
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)
        # 双流 Transformer 块列表
        self.transformer_blocks = nn.LayerList([
            FluxTransformerBlock(dim=self.inner_dim, 
                                  num_attention_heads=num_attention_heads,
                                  attention_head_dim=attention_head_dim)
            for _ in range(num_layers)
        ])
        # 单流 Transformer 块列表
        self.single_transformer_blocks = nn.LayerList([
            FluxSingleTransformerBlock(dim=self.inner_dim, 
                                       num_attention_heads=num_attention_heads,
                                       attention_head_dim=attention_head_dim)
            for _ in range(num_single_layers)
        ])
        # 输出层归一化和线性投射，将inner_dim投射回图像patch的像素维度
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, ...)
        self.proj_out = nn.Linear(self.inner_dim, patch_size*patch_size*out_channels)
```
以上代码展示了模型构造的关键部分：
* self.inner_dim 是 Transformer 隐藏维度（例如 3072），通常由注意力头数 × 每头维度计算。
* FluxPosEmbed 实例用于生成旋转位置嵌入（Rotary Positional Embedding），适用于二维图像patch网格的位置编码。
* time_text_embed 是时间步嵌入与文本全局嵌入的融合模块。
* context_embedder 是一个线性层，用于将文本编码器输出的上下文序列（如T5文本encoder输出，尺寸joint_attention_dim=4096）投射到Transformer内部维度。换言之，文本每个Token的高维embedding将降维为与图像latent相同的维度，以便进入Transformer的注意力计算。
* x_embedder 是一个线性层，将输入的图像latent通道数 (in_channels，默认64) 转为 inner_dim。FLUX模型直接在VAE生成的latent上应用Transformer，因此首先用全连接把latent特征映射到Transformer的隐藏维度。
* transformer_blocks 列表包含 num_layers 个 FluxTransformerBlock，即双流Transformer块。默认19层，用于同时处理图像和文本两个流。
* single_transformer_blocks 列表包含 num_single_layers 个 FluxSingleTransformerBlock，即单流Transformer块。默认38层，用于仅处理图像流。
* 最后，通过 norm_out（持续型AdaLN归一化）和 proj_out 输出线性层，将Transformer输出变换回原始latent形状。

下面我们深入FluxTransformerBlock（双流块）和 FluxSingleTransformerBlock（单流块）的实现细节，看看双流和单流块在注意力和前馈层上的差异。

### FluxTransformerBlock模块
双流块承担跨模态融合的任务，每层同时更新图像隐藏状态和文本隐藏状态。FluxTransformerBlock内部包含两套并行的子层：一套针对图像hidden_states，另一套针对文本encoder_hidden_states。每套都包括AdaLayerNorm（带时序嵌入调制的层归一化）、多头注意力、和前馈网络(FeedForward)，但注意力层是共享的，实现图像-文本的交互。其构造如下：
```python
class FluxTransformerBlock(nn.Layer):
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()
        # AdaLayerNorm，用Zero初始化偏置和增益，用于图像流
        self.norm1 = AdaLayerNormZero(dim)
        # AdaLayerNorm，用于文本流
        self.norm1_context = AdaLayerNormZero(dim)
        # 多头注意力层（图像-文本双流交互）
        self.attn = Attention(
            query_dim=dim,         
            cross_attention_dim=None, 
            added_kv_proj_dim=dim, 
            dim_head=attention_head_dim, 
            heads=num_attention_heads,
            out_dim=dim, 
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor2_0(),  
            qk_norm=qk_norm, 
            eps=eps,
        )
        # 图像流的LayerNorm + FeedForward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff    = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # 文本流的LayerNorm + FeedForward
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context    = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
```
双流块初始化时，设置了两套归一化和前馈，但共享一个注意力层：
* AdaLayerNormZero 是一种可调制的层归一化，初始时scale和bias为0，但在前向过程中将利用时间步嵌入提供的参数对归一化后的张量施加缩放和平移，以及门控系数。这里分别对图像(norm1)和文本(norm1_context)各用一个 AdaLN。
* Attention 层参数设置比较关键：query_dim=dim且added_kv_proj_dim=dim。这意味着查询Q来自图像hidden_states，键K/值V除了处理Q自身（图像）的投影外，还会额外投影处理一个维度为dim的输入——也就是我们传入的文本encoder_hidden_states。这相当于在一次注意力计算中同时让图像特征关注自身（自注意力）和文本特征（交叉注意力）。因为cross_attention_dim=None且用了added_kv_proj_dim=dim，所以实现上采用单一Attention来处理联合的KV。context_pre_only=False表明并非纯先验context模式，而是正常交互。总的来说，attn层实现了一个“双流融合注意力”：Queries是图像token，Keys/Values是图像token和文本token的混合。
* norm2/norm2_context 和 ff/ff_context 则是标准Transformer的后半部分（LayerNorm+前馈MLP），分别应用于图像流和文本流。注意它们没有Ada前缀，表示这些层不直接受时间嵌入调制。

FluxTransformerBlock的 forward 方法更能体现双流机制。关键步骤如下：
```python
def forward(
    self,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    temb: paddle.Tensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    # AdaLayerNormZero：归一化 & 提取门控/偏移参数
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}
    # 多头注意力：将规范化后的图像&文本特征输入注意力层
    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    # Attention输出可能包含2或3个张量
    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    # 将注意力输出应用Gate并加入残差
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    
    # 前馈层 (图像流)
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + ip_attn_output

    # 前馈层 (文本流)
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    # FP16溢出处理（剪裁），然后返回两个更新后的hidden_states
    if encoder_hidden_states.dtype == paddle.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states
```
这段逻辑实现了图像-文本双流Transformer层的前向计算，流程可总结如下：
* AdaLayerNorm 调制：使用当前扩散时间步的嵌入，对图像和文本两个输入分别做AdaLN归一化。AdaLayerNormZero 不仅返回归一化后的张量，还提取出若干调制参数：对于每个流，产生用于注意力输出的门控系数gate_msa、用于MLP输入的仿射变换参数scale_mlp 和 shift_mlp、以及MLP输出的门控系数gate_mlp。这些参数形状一般是 [batch, dim] 或 [batch,]，后续用来调整该层的输出幅度。
* 联合注意力计算：将归一化后的图像特征 norm_hidden_states 作为 Query，文本特征 norm_encoder_hidden_states 作为附加的KV输入，喂给 self.attn。image_rotary_emb 则是由 FluxPosEmbed 生成的旋转位置嵌入参数（正弦/余弦基），用于在Attention内部对Q,K应用位置编码，特别适用于图像patch的2D位置信息。
* 残差连接（注意力层）：对注意力结果施加AdaLN提供的门控系数，然后加回各自的残差通道。这样图像特征经过自注意力并融合了文本信息，文本特征也在交互中得到更新（捕获与图像的关联）。
* 前馈层 (图像)：对更新后的 hidden_states 再做一层标准LayerNorm，然后利用AdaLN的 scale 和 shift 参数对其按元素缩放和平移。随后通过 FeedForward MLP（两层全连接+激活，已经在构造时定义）变换得到 ff_output。再乘以 gate_mlp（AdaLN提供的MLP门控系数）进行缩放，最后残差连接加回到 hidden_states。
* 前馈层 (文本)：类似地，对文本 encoder_hidden_states 应用LayerNorm、AdaLN缩放偏移，再经过文本流自己的 ff_context MLP，乘以 c_gate_mlp 后加回。这样文本特征也通过MLP得到更新。
* 返回结果：最终，该层输出更新后的 encoder_hidden_states（文本）和 hidden_states（图像）。这两个将作为下一层 FluxTransformerBlock 的输入，实现逐层交替强化图像和文本的表示。注意在FP16情况下对结果裁剪，以避免数值溢出。

双流Transformer块每层都让图像和文本特征互相融合：图像latent通过交叉注意力“看”文本embedding，文本embedding也被图像特征影响更新。这类似 Stable Diffusion 3的双流Transformer设计。Flux将若干这样的层堆叠，使得高层的图像特征已深度融合文本语义。

### FluxSingleTransformerBlock模块
经过多层双流块后，FluxTransformer2DModel后半部分采用单流Transformer块，此时文本特征不再更新，仅作为条件参与图像Transformer。FluxSingleTransformerBlock与双流块的区别在于：它只维护图像一个流，并且将注意力和前馈合并为一个更紧凑的结构。下面是 FluxSingleTransformerBlock 的主要实现：
```python
class FluxSingleTransformerBlock(nn.Layer):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        # 单流AdaLN（只返回一个门控参数）
        self.norm = AdaLayerNormZeroSingle(dim)
        # 简化的MLP：先线性扩张，再激活
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        # 输出线性：将 [attn输出 + mlp输出] 合并回 dim
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        # 注意力层：不引入额外context，pre_only=True用于优化
        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim, cross_attention_dim=None, 
            dim_head=attention_head_dim, heads=num_attention_heads,
            out_dim=dim, bias=True,
            processor=processor, qk_norm="rms_norm", eps=1e-6,
            pre_only=True
        )
    def forward(
        self,
        hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        # AdaLayerNormSingle：归一化 + 提取单一门控参数
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        # 将注意力输出和MLP输出拼接，然后通过线性层融合
        hidden_states = paddle.concat([attn_output, mlp_hidden_states], axis=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        # FP16剪裁
        if hidden_states.dtype == paddle.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states
```
单流块仍然利用 AdaLN（但Simple版）对特征进行归一化和调制，但内部流程相较双流块有几点不同：
* 没有文本 context 输入：Attention层的 cross_attention_dim=None 且不使用 added_kv_proj_dim，因此这个注意力就是标准自注意力，仅作用于图像latent序列自身。也就是说，从单流块开始，模型不再更新文本embedding，文本提供的条件已经在双流块融合完毕。
* AdaLayerNormZeroSingle 返回的不是五个参数，而是norm后的hidden和一个门控系数gate。Single版的AdaLN对结构进行了简化，因为此时我们不再需要分别对注意力和MLP输出进行不同门控（它直接把二者concat后一起门控）。
* 前馈合并简化：Single块中，将注意力输出和MLP输出拼接后一起投射，然后通过 proj_out 线性层将拼接后的向量映射回 dim长度，再乘以AdaLN提供的 gate 系数，最后加上残差。
* 这样的设计实质上等效于Transformer中的并行FFN和Attention路径，只是这里不是先后顺序叠加，而是并联后融合。

单流块执行标准Transformer对图像latent的自注意力和前馈，但由于文本信息已嵌入，它不显式处理文本。Single块采用更紧凑的并行融合方式，最终继续细化图像latent。


### Prompt Embedding融合机制
FluxPipeline需要将用户输入的文本提示经过两个文本编码器（CLIP 和 T5）得到embedding，然后送入FluxTransformer2DModel。该过程涉及两个方面：
* 文本序列Embedding（prompt_embeds）：由 T5 编码器输出，包含文本每个token的上下文表示（[batch, seq_len, 4096]），用于双流块中的文本流 encoder_hidden_states。在进入FluxTransformer2DModel前，这个高维序列会通过 self.context_embedder 降维到 inner_dim（如3072）。
* 文本全局Embedding（pooled_prompt_embeds）：由 CLIP 文本模型输出，比如取 [EOS] token 的隐藏态作为整体语义表示（[batch, 768]）。这相当于一句话的语义句向量，供模型全局调控使用。

FLUX 将扩散时间步和上述全局文本embedding融合为一个向量，用于调制Transformer层。这由CombinedTimestepTextProjEmbeddings 模块完成。其代码表示如下：
```python
class CombinedTimestepTextProjEmbeddings(nn.Layer):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning
```
CombinedTimestepTextProjEmbeddings将两种不同来源的embedding简单逐元素相加。这样产生的输出向量既包含当前扩散步骤的信息，也包含了与提示文本内容相关的全局语义。这个 temb 将在Transformer每层的 AdaLayerNorm 中使用，从而影响模型中不同层的归一化和门控参数，实现条件控制。

在Prompt Embedding 生成流程中，当用户提供 prompt 文本时，FluxPipeline会用CLIP的tokenizer和文本编码器编码一次，用T5的tokenizer和编码器编码一次，代码表示如下：
```python
def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[paddle.Tensor] = None,
    pooled_prompt_embeds: Optional[paddle.Tensor] = None,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
    text_ids = paddle.zeros([prompt_embeds.shape[1], 3]).astype(dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids
```
包含两个过程：
* CLIP编码：得到 pooled_prompt_embeds，即CLIP文本模型的池化输出。
* T5编码：得到 prompt_embeds，即T5编码器最后一层隐状态序列（长度等于文本token数）。

随后，Pipeline会对这些embedding进行扩展维度，然后调用 FluxTransformer2DModel 时传入。通过这种双编码器方案，Flux模型结合了丰富的文本上下文（T5大模型提供更长更详细的语义序列）和强语义对齐的句向量（CLIP提供图文对齐的embedding）。前者用于细粒度引导图像生成（通过cross-attention作用于每层latent），后者用于全局风格/内容控制（通过temb影响模型层参数）。这种策略在 Stable Diffusion XL 中也有体现，能提升文本到图像生成的表现。

### FlowMatchEulerDiscreteScheduler采样
FLUX 使用 FlowMatchEulerDiscreteScheduler 作为扩散采样器。这个调度器基于Euler离散解算方法，并结合了Flow Matching思想。下面我们讲解其时间步设置和单步更新公式。
时间步与sigma设置
不同于DDIM或PNDM这类按固定 $\alpha$ 衰减步长的调度器，FlowMatchEulerDiscreteScheduler采用了sigma序列（噪声标准差）来表示时间。其包含 set_timesteps 函数，主要生成一串从高噪声到低噪声的sigma列表。例如，默认训练扩散步数 num_train_timesteps=1000，sigma通常从接近1降到0。其主要代码如下：
```python
def set_timesteps(
    self, 
    num_inference_steps: int = None,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
):
    if sigmas is None:
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )

        sigmas = timesteps / self.config.num_train_timesteps
    else:
        sigmas = np.array(sigmas).astype(np.float32)
        num_inference_steps = len(sigmas)
    self.num_inference_steps = num_inference_steps

    if self.config.use_dynamic_shifting:
        sigmas = self.time_shift(mu, 1.0, sigmas)
    else:
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
    
    timesteps = np.linspace(
        self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
    )

    sigmas = paddle.to_tensor(sigmas).astype(dtype=paddle.float32)
    timesteps = sigmas * self.config.num_train_timesteps

    if self.config.invert_sigmas:
        sigmas = 1.0 - sigmas
        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = paddle.concat([sigmas, paddle.ones(1)])
    else:
        sigmas = paddle.concat([sigmas, paddle.zeros(1)])

    self.timesteps = timesteps
    self.sigmas = sigmas
    self._step_index = None
    self._begin_index = None
```
调度器支持动态调整sigma序列，以适应不同图像分辨率或所需风格：
* shift（默认1.0）控制sigma时间的平移/形变。如果shift != 1，会对原始sigma值进行非线性挤压或拉伸。当shift>1时，高sigma段被压缩，低sigma段拉长，反之亦然。这样可以控制生成图像的多样性和稳定性（shift大→变化快，细节丰富；shift小→平滑收敛）。
* 动态 shifting：如果 use_dynamic_shifting=True，则需要传入参数 mu 表征当前图像分辨率相对于base分辨率的尺度。调度器会调用 self.time_shift(mu, 1.0, sigmas) 根据图像token数量在 base_image_seq_len 与 max_image_seq_len 范围内对sigmas做指数或线性偏移（由 time_shift_type 控制）。这意味着高分辨率图像可能使用不同的sigma演化速度，以稳定生成。默认该功能关闭。

最终，set_timesteps 生成一个长度为 num_inference_steps 的sigma张量（并在最后附加一个终值，用于终止计算）。对应的 self.timesteps 则是 sigma * num_train_timesteps，等价于实际扩散时间t。我们在Pipeline调用 scheduler.set_timesteps() 后，就获得了下降的噪声调度序列。

单步Euler更新公式
FlowMatchEulerDiscreteScheduler采用一阶Euler方法求解扩散逆过程的微分方程。每一步的更新逻辑在其 step 函数中实现，简化过程如下：
```python
def step(
    self,
    model_output: paddle.Tensor,
    timestep: Union[float, paddle.Tensor],
    sample: paddle.Tensor,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    if self.step_index is None:
        self._init_step_index(timestep)

    # 将当前latent sample提升为float32计算以确保精度
    sample = sample.cast(paddle.float32)
    
    # 取当前和下一步对应的 sigma 值
    sigma = self.sigmas[self.step_index]
    sigma_next = self.sigmas[self.step_index + 1]

    # Euler更新: x_{t_next} = x_t + dt * model_output
    prev_sample = sample + (sigma_next - sigma) * model_output
    prev_sample = prev_sample.cast(model_output.dtype)

    # 更新步计数
    self._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
```
这里，model_output 通常是扩散模型预测的某种噪声或变化。在标准扩散中，模型通常预测噪声 $\epsilon_\theta(x_t, t)$ ，而Euler更新实际上在模拟解 $dx/dt$ ，可以简单把 model_output 当作给定当前状态的变化量。由于 $sigma_{next} < sigma_t$ （噪声水平递减），所以 $dt$ 为负值， $prev_sample = sample + dt * model_output$ 实际相当于减去一定比例的噪声，从而逐步去噪。这和DDIM解算类似，只是这里步长是线性近似而非严格解公式。

FlowMatchEulerDiscreteScheduler之所以命名为FlowMatch，因为它结合了Flow Matching方法。然而，从代码层面我们可以把它当作Euler离散采样，加上一些sigma序列的技巧，用法上与普通Euler调度类似。总之，FluxPipeline在每个扩散步都会调用 scheduler.step(model_output, t, latents) 以得到上一时刻的latent。这一过程持续迭代，直到达到最小sigma（接近0噪声）完成去噪。

## 四、上手教程
我们以FLUX.1-dev为例，在单卡A100上进行效果演示，需要40G显存。

在安装好paddlepaddle-gpu后，执行以下shell脚本安装PPDiffusers：
```
# clone PaddleMIX代码库
git clone https://github.com/PaddlePaddle/PaddleMIX.git

cd PaddleMIX/ppdiffusers
pip install -e . --user
pip install -r requirements.txt --user
```



### 文生图
运行以下命令即可根据一段文本生成相应的图片：
```python
import paddle

from ppdiffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16, low_cpu_mem_usage=True, map_location="cpu",
)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(42)
).images[0]
image.save("text_to_image_generation-flux-dev-result.png")
```

图片生成效果如下：

<div align="center">
    <img src="https://github.com/user-attachments/assets/c3f81586-7f2e-4c91-ad0e-f2aa83998564" alt="Image 2" style="width: 60%;">
    <p style="color: #808080;"> FluxPipeline生成图片 </p>
</div>


### 图生图
运行以下命令即可根据一段文本和一张底图生成相应的图片：
```python
import paddle
from ppdiffusers import FluxImg2ImgPipeline
from ppdiffusers.utils import load_image

pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16, low_cpu_mem_usage=True, map_location="cpu",
)


url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(
    height=512,
    width=768,
    prompt=prompt, image=init_image, num_inference_steps=50, strength=0.95, guidance_scale=0.0
).images[0]

images.save("text_to_image_generation-flux-dev-result_img2img.png")
```

图片生成效果如下：

<div align="center">
    <img src="https://github.com/user-attachments/assets/4257a068-8cdb-467c-901d-9f21123cf010" alt="Image 2" style="width: 60%;">
    <p style="color: #808080;"> FluxImg2ImgPipeline输入图片 </p>
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/67d8cdd8-a495-4b10-bf1d-a9ea8b674107" alt="Image 2" style="width: 60%;">
    <p style="color: #808080;"> FluxImg2ImgPipeline生成图片 </p>
</div>

五、总结
在跨模态生成技术领域，FLUX引入了全新架构，显著提升了图像生成的质量。百度飞桨团队推出的PaddleMIX套件现已完整实现推理全流程支持。通过深入分析其代码实现，研究人员和开发者能够更清晰地掌握模型的核心技术细节与创新要点。
FLUX 是由 Black Forest Labs 开发的文本到图像生成模型，采用混合多模态和并行扩散 Transformer 架构，结合流匹配技术，旨在提升生成图像的质量和效率。 