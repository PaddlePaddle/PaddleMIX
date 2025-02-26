# 前沿多模态模型开发与应用实战第二期：Qwen2-VL系列多模态理解大模型算法解析与功能抢先体验

多模态理解大模型是一类能够同时处理和理解多种数据形式（如图像📸、文本📝、视频🎥等）的人工智能模型。这类模型通过深度学习技术，可以实现跨模态的信息理解、关联和生成！相比传统的单模态模型，多模态模型能够更全面地理解和分析复杂场景，在实际应用中具有更强的实用性和普适性。✨典型应用包括：图文理解、视觉问答、文档理解、场景描述等任务。随着技术的发展，多模态大模型在准确性、鲁棒性和通用性等方面都取得了显著进步，为人工智能的发展开辟了新的方向！🎯

本文将介绍Qwen2-VL系列模型，涵盖Qwen2-VL和Qwen2.5-VL。内容包括模型细节、训练流程、模型能力的展示，并以PaddleMIX中Qwen2.5-VL的实现为例，对代码进行逐步解读。

## 一、引言
Qwen2-VL系列是对Qwen-VL模型的改进升级，重新定义了传统的预设分辨率方法在视觉处理中的应用。Qwen2-VL引入了原生动态分辨率机制，使得模型能够动态地将不同分辨率的图像处理成不同数量的视觉tokens。该方法能够生成更高效、准确的视觉表征，更贴近人类的感知过程。模型还集成了多模态旋转位置嵌入（M-RoPE），促进了文本、图像和视频之间位置信息的有效融合。Qwen2-VL系列采用了统一的图像和视频处理范式，提升了模型的视觉感知能力。

Qwen2.5-VL是对Qwen2-VL的进一步优化，通过在图像上使用图像的实际尺寸来表示坐标，时间上引入了动态 FPS (每秒帧数)训练和绝对时间编码，将 mRoPE id 直接与时间流速对齐，改进提升了模型对时间和图像尺寸的感知。同时Qwen2.5-VL重新训练了更简单高效的视觉编码器。

<div align="center">
    <img src="https://github.com/user-attachments/assets/615b3af1-4e59-46bf-8f44-adf304976f3b" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图1 Qwen2-VL能力展示
 </p>
</div>

如图1所示，在模型能力上Qwen2-VL能够实现多语种图像文本理解、代码/数学推理、视频分析、直播聊天、智能体等。而Qwen2.5-VL在此基础上进一步能实现理解长视频和捕捉事件、图像细粒度感知，初步使用电脑手机等能力。

在模型尺寸上，Qwen2-VL系列目前有以下这些参数版本👇
Qwen2-VL：2B，7B，72B
Qwen2.5-VL：3B，7B，72B

## 二、方法
### 2.1 Qwen2-VL
<div align="center">
    <img src="https://github.com/user-attachments/assets/456e9b61-c7a9-40da-a355-07327bddf577" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图2: Qwen2-VL架构
 </p>
</div>

如图2，整体上Qwen2-VL仍然延续了 Qwen-VL 中 ViT 加 Qwen2 的串联结构，在三个不同尺度的模型上，Qwen2-VL 都采用 600M 规模大小的 ViT，并且支持图像和视频统一输入。为了让模型更清楚地感知视觉信息和理解视频，Qwen2-VL 还进行了以下升级：
* 原生动态分辨率：Qwen2-VL 在架构上的一大改进是实现了对原生动态分辨率的全面支持。与上一代模型相比，Qwen2-VL 能够处理任意分辨率的图像输入，不同大小图片被转换为动态数量的 tokens，最小只占 4 个 tokens。这种设计不仅确保了模型输入与图像原始信息之间的高度一致性，更是模拟了人类视觉感知的自然方式，赋予模型处理任意尺寸图像的强大能力，使其在图像处理领域展现出更加灵活和高效的表现。具体而言，为了减少每个图像的视觉令tokens，Qwen2-VL 在ViT之后使用一个简单的MLP层将相邻的2 × 2 tokens压缩为单个token，并将特殊的<|vision_start|>和<|vision_end|> token放置在压缩的视觉tokens的开始和结束处。因此，分辨率为224 × 224的图像，使用patch_size = 14的ViT编码，在进入LLM之前，将被压缩到66个tokens。
* 多模态旋转位置嵌入（M-ROPE）：Qwen2-VL 在架构上的另一重要创新则是多模态旋转位置嵌入（M-ROPE）。传统的旋转位置嵌入只能捕捉一维序列的位置信息，而 M-ROPE 通过将原始旋转嵌入分解为代表时间、高度和宽度的三个部分，使得大规模语言模型能够同时捕捉和整合一维文本序列、二维视觉图像以及三维视频的位置信息。这一创新赋予了语言模型强大的多模态处理和推理能力，能够更好地理解和建模复杂的多模态数据。

<div align="center">
    <img src="https://github.com/user-attachments/assets/7b6acde5-7257-458b-827a-e1e37295719e" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图3: 多模态旋转位置嵌入（M-ROPE）示意图
 </p>
</div>

* 统一图像和视频理解：Qwen2-VL采用图像和视频数据混合的训练方案，保证了图像理解和视频理解的充分性。为了尽可能完整地保留视频信息，Qwen2-VL以每秒两帧的速度对每个视频进行采样。此外，Qwen2-VL集成了深度为2的3D卷积来处理视频输入，使模型能够处理3D管道而不是2D补丁，从而使其能够在不增加序列长度的情况下处理更多的视频帧。为了保持一致，每幅图像被视为两个相同的帧。为了平衡长视频处理的计算需求和整体训练效率，Qwen2-VL 动态调整每个视频帧的分辨率，将每个视频的tokens总数限制为16384个。这种训练方式在模型对长视频的理解能力和训练效率之间取得了平衡。

## 2.2 Qwen2.5-VL

<div align="center">
    <img src="https://github.com/user-attachments/assets/0710d2c1-3bd3-4acd-b394-66da569f3d57" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图4: Qwen2.5-VL架构 </p>
</div>

Qwen2.5-VL在整体架构上没有太大变动，主要涉及以下两个改进：
* 时间和图像尺寸的感知：在空间维度上，Qwen2.5-VL 不仅能够动态地将不同尺寸的图像转换为不同长度的 token，还直接使用图像的实际尺寸来表示检测框和点等坐标，而不进行传统的坐标归一化。这使得模型能够直接学习图像的尺度。在时间维度上，引入了动态 FPS (每秒帧数)训练和绝对时间编码，将 mRoPE id 直接与时间流速对齐。这使得模型能够通过时间维度 id 的间隔来学习时间的节奏。
* 更简洁高效的视觉编码器：视觉编码器在多模态大模型中扮演着至关重要的角色。Qwen2.5-VL从头开始训练了一个原生动态分辨率的 ViT，包括 CLIP、视觉-语言模型对齐和端到端训练等阶段。为了解决多模态大模型在训练和测试阶段 ViT 负载不均衡的问题，Qwen2.5-VL引入了窗口注意力机制，有效减少了 ViT 端的计算负担。在Qwen2.5-VL的 ViT 设置中，只有四层是全注意力层，其余层使用窗口注意力。最大窗口大小为 8x8，小于 8x8 的区域不需要填充，而是保持原始尺度，确保模型保持原生分辨率。此外，为了简化整体网络结构，Qwen2.5-VL使 ViT 架构与 LLMs 更加一致，采用了 RMSNorm 和 SwiGLU 结构。

## 三、训练方法
### 3.1 训练pipeline
遵循Qwen-VL，Qwen2-VL采用了三阶段的训练方法。在第一阶段，Qwen2-VL专注于训练ViT组件，利用大量的图像-文本对来增强大语言模型中的语义理解。在第二阶段，Qwen2-VL解冻所有参数，并使用更广泛的数据进行训练，以进行更全面的学习。在最后阶段，Qwen2-VL冻结ViT参数，并使用指导数据集对LLM进行特定的微调。其中第一、第二阶段实际都是在预训练阶段，第三阶段是指令微调阶段。

第一阶段：在初始预训练阶段，Qwen2-VL使用了约6千亿个tokens的语料进行训练。Qwen2-VL的LLM部分采用了Qwen2 的参数初始化，而视觉编码器则基于DFN的ViT进行初始化。然而，DFN的ViT中的固定位置嵌入被RoPE-2D所替代。该预训练阶段主要致力于学习图像与文本之间的关系，特别是通过识别图像内部的文本内容OCR和图像分类任务作为基础训练，帮助模型在视觉-文本关系的学习中打下坚实基础。这种训练方式使Qwen2-VL能够在处理视觉和文本数据时，稳定地理解并对齐图像中的关键信息与文本描述。通过这种方式，模型在面对更复杂的任务时，能更准确地捕捉到图像和文本之间的内在联系，为后续的微调和特定任务的执行提供强有力的支持。
第二阶段：第二个预训练阶段标志着Qwen2-VL训练的重大进步，该阶段增加了额外的8千亿tokens的图像相关数据。此阶段引入了更多图文混合内容，帮助模型更细致地理解视觉与文本信息之间的复杂相互作用。视觉问答数据集的引入进一步细化了模型对图像相关查询的响应能力，使其能够更准确地回答与图像内容相关的问题。此外，多任务数据集的加入增强了模型在不同任务之间的并发导航能力，这是在处理复杂且多样化的现实世界数据集时的关键优势。同时，纯文本数据的继续融入，确保了模型语言理解和生成的高水平精确度，进一步提升了其在处理语言任务中的表现。

在整个预训练阶段，Qwen2-VL累计处理了1.4万亿tokens。这些token不仅包括文本标记，还涵盖了图像标记。然而，在训练过程中，模型仅对文本标记进行监督。这种对广泛且多样的语言与视觉场景的深入接触，使得Qwen2-VL能够全面理解视觉和文本信息之间错综复杂的关系。这为模型在处理各种多模态任务时提供了坚实的基础，确保了其能够在面对实际应用中的复杂任务时展现出强大的能力和灵活性。

第三阶段：在指令微调阶段，Qwen2-VL采用了ChatML的格式来构建指令跟随数据集。该数据集不仅包括纯文本对话数据，还融合了多模态对话数据。多模态数据包括图像问答、文档解析、多图像比较、视频理解、视频流对话和基于Agent的交互等。通过这种多维度的数据构建方法，Qwen2-VL旨在增强模型理解并执行各种模态下的广泛指令的能力。融合不同的数据类型，使Qwen2-VL能够应对复杂的多模态任务，而不局限于传统的文本交互。这样，模型在面对现实世界中丰富的多样化场景时，能够展现出更强的适应性和鲁棒性，为各种实际应用提供更加精准的支持。

与Qwen-VL一样，Qwen2-VL也使用了特殊的token来区分视觉和文本输入。在图像特征序列的开始和结束处插入特殊的\<|vision_start|>和\<|vision_end|> token，用于标定图像内容。
* 对话数据：在对话格式方面，Qwen2-VL使用ChatML格式构建了指令调优数据集，其中每个交互的语句都用两个特殊的token (<|im_start|>和<|im_end|>)来标记，以方便对话的终止。蓝色标记的部分表示被监督的部分。

<div align="center">
    <img src="https://github.com/user-attachments/assets/1f3fa79b-a052-4180-9bf4-a712ea5b3c5e" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图5: Chatml的Dataset格式示例 </p>
</div>

* 视觉定位：为了使模型具备视觉定位能力，Qwen2-VL将边界框坐标归一化至\[0, 1000\)范围，并表示为$(X_{top-left},Y_{top-left})$ 和$(X_{bottom-right},Y_{bottom-right})$ 。这些坐标与文本一同作为tokens进行处理，用于标注边界框文本。为了精确地将边界框与其文本描述对应，Qwen2-VL引入了\<|object_ref_start|>和\<|object_ref_end|> token，明确指出每个边界框所对应的具体内容，从而使模型能够有效地理解并生成对特定区域的准确描述。
<div align="center">
    <img src="https://github.com/user-attachments/assets/c0c877fe-6d27-4f29-9553-963e6b55673c" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图6: 指示性定位数据格式示例 </p>
</div>
* 视觉Agent：为了将Qwen2-VL发展为通用的VL-Agent，Qwen2-VL将各种Agent任务（如UI操作、机器人控制、游戏、导航等）视为序列决策问题，使Qwen2-VL能够通过多步动作执行完成任务。针对每个任务，Qwen2-VL首先为函数调用定义了一组允许的动作和关键字模式（以下划线表示）。接着，Qwen2-VL分析观察到的环境信息，进行推理和规划，选择并执行相应的动作，与环境互动以获取新的观察数据。此过程会循环进行，直到任务顺利完成。通过集成各种工具，并利用大型视觉语言模型的视觉感知能力，Qwen2-VL能够进行增量式的迭代执行。
<div align="center">
    <img src="https://github.com/user-attachments/assets/c0c877fe-6d27-4f29-9553-963e6b55673c" alt="Image 2" style="width: 100%;">
    <p style="color: #808080;"> 图7: 视觉Agent数据格式示例 </p>
</div>


## 四、代码解读

下面以PaddleMIX中Qwen2.5-VL的实现为例，对Qwen2-VL中关键创新点的代码实现进行讲解。
### 3.1 视觉Token压缩
这是Qwen2-VL相对于Qwen-VL的重要改进点，通过token合并减少视觉tokens的数量。在具体实现上主要涉及**Qwen2_5_VLPatchMerger**类

**功能说明**:

该类用于将视觉特征图中的patches进行合并和维度转换。它主要用于视觉编码器中,通过合并相邻的patch特征来减少序列长度并调整特征维度。

**实现细节**:
```python
class Qwen2_5_VLPatchMerger(paddle.nn.Layer):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        # 计算合并后的隐藏维度
        self.hidden_size = context_dim * (spatial_merge_size**2)
        # 层归一化
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        # MLP层用于特征转换
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # 对输入进行归一化并重塑维度
        x = self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))
        return x
```

**参数说明**:
* **dim**: 输出特征的维度
* **context_dim**: 输入特征的维度
* **spatial_merge_size**: 空间合并的大小,默认为2(即2x2的patch会被合并)

**调用示例**:
```python
# 假设有视觉特征 patches
# patches shape: [batch_size, num_patches, context_dim]
patch_merger = Qwen2_5_VLPatchMerger(
    dim=1024,           # 输出维度
    context_dim=768,    # 输入维度
    spatial_merge_size=2  # 2x2合并
)

# 前向传播
merged_features = patch_merger(patches)
# merged_features shape: [batch_size * (num_patches/4), 1024]
```

处理流程:
1. 输入特征首先经过RMSNorm层归一化
2. 将归一化后的特征重塑为合并后的维度 (context_dim * spatial_merge_size²)
3. 通过两层MLP网络进行特征转换:
  * 第一层Linear保持维度不变
  * 中间使用GELU激活函数
  * 第二层Linear将维度转换为目标维度dim
4. 输出转换后的特征向量

这个模块在视觉编码器中起到了降低序列长度、调整特征维度的重要作用,有助于提高模型的计算效率和特征表达能力。

### 3.2 多模态旋转位置嵌入（M-ROPE）
多模态旋转位置编码是Qwen2-VL的关键创新之一,它通过3D位置编码增强了模型的空间感知能力。对于视觉内容,分别在时间、高度和宽度三个维度应用RoPE,使模型能更好地理解视觉内容的空间结构；对于文本内容,则自动退化为传统的1D RoPE,与现代LLM保持一致。这种统一的编码框架不仅提供了高效的位置信息注入方式,还促进了视觉-文本的深度融合,是模型在多模态理解和生成任务上取得优异性能的重要基础。

**功能说明**:
该函数实现了多模态3D旋转位置编码,是对传统1D RoPE的扩展。主要用于处理视觉(图像/视频)和文本的混合输入序列的位置编码。

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # 将section尺寸翻倍(因为每个维度需要cos和sin两部分)
    mrope_section = mrope_section * 2
    
    # 处理cos部分
    cos = paddle.concat(
        x=[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], 
        axis=-1
    ).unsqueeze(axis=unsqueeze_dim)
    
    # 处理sin部分
    sin = paddle.concat(
        x=[m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], 
        axis=-1
    ).unsqueeze(axis=unsqueeze_dim)

    # 应用旋转位置编码
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**创新点**:
1. **3D位置编码**:
  * 对视觉部分,分别在时间(temporal)、高度(height)和宽度(width)三个维度应用RoPE
  * 将channel维度分成3个chunk,分别对应这三个空间维度
  * 文本部分仍使用传统1D RoPE,保持与现代LLM一致
2. **统一的编码框架**:
  * 同时处理视觉和文本的位置信息
  * 对文本部分,三个维度的位置索引相同,退化为标准RoPE
  * 实现了视觉-文本位置编码的无缝集成
**实现细节**:
1. **维度分割**:
```python
mrope_section = mrope_section * 2  # 因为每个维度需要cos和sin
```
2. **循环应用**:
```python
[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))]
```


* 将特征分成多个section
* 使用i % 3循环应用三个维度的位置编码

3. **位置编码应用**:
```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```
* 使用复数运算的等价形式应用RoPE
* rotate_half函数实现特征向量的半旋转

**调用示例**:
```python
# 在Qwen2_5_VLAttention层中的使用案例
class Qwen2_5_VLAttention(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # ... 初始化参数 ...
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ):
        # ... 获取batch_size和序列长度 ...

        # 1. 生成QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 2. 重塑和转置
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = query_states.reshape([0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([0, 0, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([0, 0, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])

        # 3. 应用多模态旋转位置编码
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # 4. 处理KV缓存
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # 5. 计算注意力并输出
        # ... 注意力计算和输出处理 ...
        
        return attn_output, attn_weights, past_key_value
```

### 3.3 更简洁高效的视觉编码器
本文在上面提到过：Qwen2.5-VL从头开始训练了一个原生动态分辨率的 ViT。为了解决多模态大模型在训练和测试阶段 ViT 负载不均衡的问题，Qwen2.5-VL引入了窗口注意力机制，有效减少了 ViT 端的计算负担。在Qwen2.5-VL的 ViT 设置中，只有四层是全注意力层，其余层使用窗口注意力。最大窗口大小为 8x8，小于 8x8 的区域不需要填充，而是保持原始尺度，确保模型保持原生分辨率。此外，为了简化整体网络结构，Qwen2.5-VL使 ViT 架构与 LLMs 更加一致，采用了 RMSNorm 和 SwiGLU 结构。具体实现涉及Qwen2_5_VisionTransformerPretrainedModel类。
**功能说明**:
该类实现了Qwen2.5-VL的视觉编码器，是一个原生动态分辨率的ViT模型。它通过窗口注意力机制和特征合并来提高计算效率，同时保持了强大的视觉理解能力。
**主要参数**:
* spatial_merge_size: 空间合并的大小，用于特征降采样
* patch_size: 图像patch的大小，用于初始特征提取
* fullatt_block_indexes: 使用全注意力的层索引
* window_size: 窗口注意力的大小，默认为8x8
* hidden_size: 隐藏层维度核心设计:

1. **窗口注意力机制**:
```python
def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor):
    # ... 
    for layer_num, blk in enumerate(self.blocks):
        # 只有特定层使用全注意力，其他使用窗口注意力
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens  # 全注意力
        else:
            cu_seqlens_now = cu_window_seqlens  # 窗口注意力
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)
```

2. **动态分辨率处理**:
```python
def get_window_index(self, grid_thw):
    # 计算窗口索引，支持动态分辨率
    vit_merger_window_size = (self.window_size // self.spatial_merge_size // self.patch_size)
    for grid_t, grid_h, grid_w in grid_thw:
        # 小于8x8的区域不填充，保持原始尺度
        llm_grid_h, llm_grid_w = (grid_h // self.spatial_merge_size, grid_w // self.spatial_merge_size)
        # ...
```
3. **位置编码优化**:
```python
def rot_pos_emb(self, grid_thw):
    # 实现高效的旋转位置编码
    # 分别处理高度和宽度维度
    hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
    wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
    # ...
```

处理流程:
1. 特征提取阶段:
    * 将输入图像划分为固定大小的patches
    * 通过patch嵌入层将patches转换为特征向量
    * 生成二维网格位置信息用于后续位置编码
2. 位置编码阶段:
    * 基于图像的高度和宽度生成网格坐标
    * 计算旋转位置编码
    * 将位置信息与特征向量对齐
3. 窗口划分阶段:
    * 根据配置的窗口大小划分特征图
    * 生成窗口索引和序列长度信息
    * 对小于窗口大小的区域保持原始尺度
4. 特征处理阶段:
    * 重排序特征以适应窗口注意力计算
    * 根据层索引选择全注意力或窗口注意力
    * 通过多层Transformer块进行特征转换
5. 特征整合阶段:
    * 通过patch合并器降低特征维度
    * 恢复特征的原始顺序
    * 输出最终的视觉特征表示
调用示例:
```python

# 初始化模型
config = Qwen2_5_VLVisionConfig(
    spatial_merge_size=2,
    patch_size=16,
    window_size=8,
    fullatt_block_indexes=[0, 1, 2, 3]  # 4层全注意力
)
vision_model = Qwen2_5_VisionTransformerPretrainedModel(config)

# 处理输入
batch_size = 1
image_size = 224
hidden_states = paddle.randn([batch_size, 3, image_size, image_size])
grid_thw = paddle.to_tensor([[1, image_size//16, image_size//16]])  # [T, H, W]

# 前向传播
output = vision_model(hidden_states, grid_thw)
```

## 五、上手教程
### Qwen2-VL系列在PaddleMIX里快速体验
通过解析代码我们也更深入地理解模型的实现细节和技术创新，快跟着我们的aistudio教程一起来动手实践一下吧！
aistudio教程链接：[【PaddleMIX】快速体验Qwen2-VL的多模态理解模型 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/8807257)

我们以Qwen2-VL-2B为例，在单卡V100上只需12G显存即可推理完成图像理解。
首先下载 PaddleMIX代码库：
```bash
# clone PaddleMIX代码库
git clone https://github.com/PaddlePaddle/PaddleMIX.git

cd PaddleMIX
```
安装PaddlePaddle：
```bash
# 提供三种 PaddlePaddle 安装命令示例，也可参考PaddleMIX主页的安装教程进行安装

# 3.0.0b2版本安装示例 (CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Develop 版本安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# sh 脚本快速安装
sh build_paddle_env.sh
```

```bash
# 提供两种 PaddleMIX 依赖安装命令示例

# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖、paddlenlp
python -m pip install -e . --user
python -m pip install -e ppdiffusers --user
python -m pip install -r requirements.txt --user
python -m pip install paddlenlp==3.0.0b3 --user

# sh 脚本快速安装
sh build_env.sh
```

## 图像理解：
运行以下命令即可：
```bash
# Qwen2.5-VL understanding
python paddlemix/examples/qwen2_5_vl/single_image_infer.py \
    --model_path="Qwen/Qwen2.5-VL-3B-Instruct" \
    --image_file="applications/MULLM/examples/haizeiwang.jpeg" \
    --question="请描述这个动漫图片，需要1. 推测动漫是哪一部；2. 给出图片的整体风格；3.描述图像中的细节，并推测可能的背景故事。" \
    --dtype="bfloat16"
```
输出结果： 
这张图片展示了一群穿着和服的角色，背景中有樱花树和蓝天白云，整体风格充满了日本传统元素和冒险气息。
```
### 1. 推测动漫是哪一部

根据角色的服装和背景，这张图片很可能来自《海贼王》（One Piece）系列。特别是图中的人物和他们的服装风格，与《海贼王》中的许多场景非常相似。

### 2. 图片的整体风格

图片的整体风格充满了日本传统和冒险的气息。角色们穿着和服，背景中有樱花树和蓝天白云，给人一种清新、明亮的感觉。这种风格通常出现在《海贼王》中的某些章节或场景中，尤其是那些描绘自然风光和角色们在海边或岛屿上的冒险经历。

### 3. 描述图像中的细节，并推测可能的背景故事

- **人物**：图片中有多个角色，包括一个穿着红色和服、手持武器的角色，可能是主角路飞；另一个穿着绿色和服、手持武器的角色，可能是索隆；还有其他几位穿着不同颜色和服的角色，可能是其他主要成员。
- **细节**：角色们的表情各异，有的微笑，有的严肃，显示出他们各自的性格特点。背景中的樱花树和蓝天白云增加了画面的美感，暗示这是一个充满希望和冒险的场景。
- **背景故事**：这张图片可能描绘的是《海贼王》中的某个重要事件，比如一次重要的战斗前的准备，或者是角色们在海边的休息时刻。樱花树和蓝天白云的背景可能象征着新的开始或希望，而角色们的服装和表情则反映了他们在面对挑战时的不同态度和情感。

总的来说，这张图片通过其独特的视觉元素和角色设计，成功地传达了《海贼王》系列特有的冒险和探索精神。
```

## 六、总结
在跨模态理解技术领域，Qwen2-VL与Qwen2.5-VL展现了突破性进展。
百度飞桨团队推出的PaddleMIX套件现已完整实现两大模型的推理全流程支持，通过深入解析其代码实现，研究人员和开发者能够更透彻地理解模型的核心技术细节与创新突破。我们诚挚推荐您访问AI Studio平台的专项教程（点击以下链接🔗），通过实践演练掌握前沿多模态模型的开发与应用技巧。

论文链接：

https://arxiv.org/pdf/2409.12191 Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution

https://arxiv.org/pdf/2502.13923 Qwen2.5-VL Technical Report

应用体验（点我试玩）：[应用中心-飞桨AI Studio星河社区](https://aistudio.baidu.com/application/detail/65916)

项目地址：

Qwen2-VL: https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/qwen2_vl

Qwen2.5-VL:https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/qwen2_5_vl

Qwen2.5-VL+R1应用:https://github.com/PaddlePaddle/PaddleMIX/tree/develop/applications/MULLM

AISTUDIO教程链接：

[【PaddleMIX】快速体验Qwen2-VL的多模态理解模型 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/8807257)