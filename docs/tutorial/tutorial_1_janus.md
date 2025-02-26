# 前沿多模态模型开发与应用实战第一期：多模态统一模型Janus解析与功能抢先体验

多模态统一模型旨在通过单一的网络结构同时处理多种模态的数据输入和输出（如文本、图像、视频等），这种模型不仅能够对图片或视频做出语义理解（如视觉问答、描述字幕等），还能根据给定文本生成高质量的图片或视频（如文生图、文生视频等）。而传统的多模态模型通常把以上两类任务分成多模态理解和多模态生成两种独立任务，并分别设计不同的结构，这样不仅增加了模型复杂度，还降低了部署效率。多模态统一模型能够高效处理混合模态任务，可以将多模态理解和多模态生成能力集成到一个统一的框架中，实现任意模态到任意模态的转换和生成。本文就将基于飞桨多模态大模型开发套件PaddleMIX介绍一下目前最热门的多模态统一模型Janus和Janus-Pro。
Janus 是 DeepSeek 团队提出的一个统一多模态理解与生成的模型，能够在单一模型中实现图像理解和文本到图像生成的双重任务。在多模态理解方面，Janus可以处理图像描述、视觉问答（VQA）、地标识别、文字识别等多种任务；在多模态生成方面，Janus也可以根据输入的文本描述生成高质量的图片。Janus-Pro是其最新的升级版本。
<div align="center">
    <img src="https://github.com/user-attachments/assets/1cb32d22-9d3b-4a23-b23c-907d45932cb5" style="width: 100%;">
</div>

Janus的核心创新点在于将多模态理解与生成的视觉编码进行解耦，从而缓解了这两个任务潜在存在的冲突。Janus-Pro在此基础上，优化训练策略（包括增加训练步数、调整数据配比等）、增加数据（包括使用合成数据等）、扩大模型规模（扩大到70亿参数），从而同时提高了模型的多模态理解和生成能力。

## Janus模型结构
Janus和Janus-Pro结构一致，均使用两个独立的编码器来理解和生成图像，而不像之前的做法依赖单个编码器来处理这两项任务。对于图像理解，Janus 使用 SigLIP 编码器将图像转换为丰富的语义特征；而对于图像生成，Janus 使用 VQ Tokenizer 将图像转换为离散标记。这种解耦的设计带来两个收益：

1）将多模态理解与生成的视觉编码解耦，缓解了多模态理解和生成不同粒度需求的冲突；

2）理解和生成任务都可以分别采用各领域最先进的编码技术，可输入其他模态例如点云或音频数据，并使用统一的Transformer进行处理。

<div align="center">
    <img src="https://github.com/user-attachments/assets/1ce50ee9-4e7c-4660-87d2-b075ca5953cd" style="width: 100%;">
    <p style="color: #808080;"> Janus结构图，“Und. Encoder”和“Gen. Encoder”分别是“理解编码器”和“生成编码器”的缩写。</p>
</div>

对于纯文本理解、多模态理解和视觉生成任务，Janus采用独立的编码方法将原始输入转换为特征，然后通过统一的自回归 Transformer 进行处理。具体来说：
* 文本理解：使用大语言模型（LLM）内置的分词器将文本转换为离散的 ID，并获取每个 ID 对应的特征表示。
* 多模态理解：使用 SigLIP 视觉编码器从图像中提取高维语义特征。这些特征从 2D 网格展平为 1D 序列，并通过一个两层MLP的理解适配器Adaptor将这些图像特征映射到 LLM 的输入空间。
* 视觉生成：使用 VQ Tokenizer将图像转换为离散的 ID。将 ID 序列展平为 1D 后，使用一个生成适配器Adaptor将每个 ID 对应的码本嵌入映射到 LLM 的输入空间。 然后，将这些特征序列连接起来，形成一个多模态特征序列，随后输入到 LLM 中进行处理。

在纯文本理解和多模态理解任务中，Janus都是使用 LLM 内置的预测头进行文本预测；而在视觉生成任务中，Janus使用随机初始化的预测头进行图像预测。整个模型是使用 Next-Token-Prediction 的方式进行训练的，采用 causal attention mask，和 LLM 的训练方式一致，遵循自回归框架。

## Janus训练流程
Janus 的训练分为三个阶段：
<div align="center">
    <img src="https://github.com/user-attachments/assets/e624f1f0-3ca1-4fbe-9aa9-8b556eacd23c" style="width: 100%;">
    <p style="color: #808080;"> Janus 三阶段训练步骤 </p>
</div>

* 第一阶段：训练Adaptor与Image Head。在嵌入空间创建语言元素与视觉元素之间的联系，使得LLM能够理解图像中的实体，并具备初步视觉生成能力； 对于多模态理解，使用来自ShareGPT-4V的125万个图像-文本配对字幕数据，格式：
\<image>\<text>； 对于视觉生成，使用来自ImageNet-1k的120万个样本，格式：\<category_name>\<image>；

* 第二阶段：统一预训练。使用多模态语料库进行统一预训练，学习多模态理解和生成。在该阶段使用纯文本数据、多模态理解数据和视觉生成数据；
  * 纯文本数据：DeepSeek-LLM预训练语料库；
  * 交错的图像-文本数据：WikiHow 和 WIT 数据集；
  * 图像Caption数据：来自多个来源的图像，并采用开源多模态模型重新为部分图像添加字幕，数据格式为问答对，如\<caption> Describe the image in detail.\<caption>；
  * 表格和图表数据：来自 DeepSeek-VL的相应表格和图表数据，数据格式为\<question><answer>；
  * 视觉生成数据：来自多个数据集的image-caption对以及 200 万个内部数据；在训练过程中，以25%的概率随机仅使用caption的第一句话；ImageNet 样本仅在最初的 120K 训练步骤中出现，其他数据集的图像在后续 60K 步骤中出现；

  * 第三阶段：监督微调。使用指令微调数据对预训练模型进行微调，以增强其遵循指令和对话的能力。微调除生成编码器之外的所有参数。在监督答案的同时，对系统和用户提示进行遮盖。为了确保Janus在多模态理解和生成方面都具备熟练度，不会针对特定任务分别微调模型。相反，Janus 使用纯文本对话数据、多模态理解数据和视觉生成数据的混合数据，以确保在各种场景下的多功能性；
    * 文本理解：使用来自特定来源的数据；
    * 多模态理解：使用来自多个来源的指令调整数据；
    * 视觉生成：使用来自部分第二阶段数据集的图像-文本对子集以及 400 万个内部数据；
    * 数据格式为：User:\<Input Message> \n Assistant: \<Response>；

## Janus-Pro性能升级与优化
Janus-Pro 是 Janus 的升级版本，它在多个方面进行了优化和改进。
* 训练策略
  * Stage 1: 增加训练步数，在 ImageNet 上充分训练；
  * Stage 2: 不再使用 ImageNet，直接使用常规text-to-image数据的训练数据；
  * Stage 3: 修改微调过程中的数据集配比，将多模态数据、纯文本数据和文本到图像的比例从 7:3:10 改为 5:1:4；
* 数据规模
  * 多模态理解
    * Stage 2: 增加 9000 万个样本，包括图像字幕数据 YFCC、表格图表文档理解数据 Doc-matrix；
    * Stage 3: 加入 DeepSeek-VL2 额外数据集，如 MEME 理解等；
  * 视觉生成：真实世界数据可能包含质量不高，导致文本到图像的生成不稳定，产生美学效果不佳的输出，Janus-Pro 使用 7200 万份合成美学数据样本，统一预训练阶段（Stage 2）真实数据与合成数据比例 1:1；
* 模型规模
  * 将模型参数扩展到 70 亿参数规模；

<div align="center">
    <img src="https://github.com/user-attachments/assets/4ffadd47-29b2-49fc-ae3a-dd0a7d30cd8b" style="width: 70%;">
    <p style="color: #808080;"> Janus-Pro训练三个阶段的超参数
 </p>
</div>

## Janus-Pro的指标结果

<div align="center">
    <img src="https://github.com/user-attachments/assets/c05d3dee-0444-4c1a-8b72-7ddb9254beb1" style="width: 70%;">
    <p style="color: #808080;"> GenEval文生图能力榜单

 </p>
</div>

## Janus 代码解析
PaddleMIX中已经复现了Janus 和 Janus-Pro 的推理流程，下面我们来具体解析一下重点代码片段。
代码目录：https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/models/janus

## Janus 组网代码
* 类名: JanusMultiModalityCausalLM
* 功能: 该类实现了一个JanusMultiModalityCausalLM模型，它能够接收图像和文本特征输入，输出为模型可以处理的嵌入向量。
  * __init__函数：
    * 从config中提取各个组件的配置。
    * 使用model_name_to_cls函数和配置参数来实例化vision_model、aligner、gen_vision_model、gen_aligner、gen_head和language_model。
  * prepare_inputs_embeds函数：
    * 重新排列图像数据pixel_values的形状以适应vision_model的输入要求。
    * 使用视觉模型处理图像数据，并通过aligner生成images_embeds。
    * 重新排列images_embeds和images_emb_mask的形状以匹配文本输入的形状。
    * 处理文本输入input_ids，将其转换为language_model可以处理的input_embeds。
    * 根据images_emb_mask将图像嵌入插入到文本嵌入中，生成最终的输入input_embeds。
  * prepare_gen_img_embeds函数：
    * 使用gen_embed层将图像标识符image_ids映射到 embedding 向量。
    * 通过aligner处理这些 embedding 向量，生成最终的图像embedding。
```python
class JanusMultiModalityCausalLM(JanusMultiModalityPreTrainedModel):
    config_class = MultiModalityConfig

    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)
        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)
        
        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)
        
        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()
        
        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)
        
        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = paddle.nn.Embedding(
            num_embeddings=gen_vision_config.params["image_token_size"],
            embedding_dim=gen_vision_config.params["n_embed"],
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: paddle.Tensor,
        pixel_values: paddle.Tensor,
        images_seq_mask: paddle.Tensor,
        images_emb_mask: paddle.Tensor,
        **kwargs
    ):
        """
        Args:
            input_ids (paddle.Tensor): [b, T]
            pixel_values (paddle.Tensor):   [b, n_images, 3, h, w]
            images_seq_mask (paddle.Tensor): [b, T]
            images_emb_mask (paddle.Tensor): [b, n_images, n_image_tokens]
            assert paddle.sum(images_seq_mask) == paddle.sum(images_emb_mask)
        Returns:
            input_embeds (paddle.Tensor): [b, T, D]
        """
        bs, n = tuple(pixel_values.shape)[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.aligner(self.vision_model(images))
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]
        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: paddle.Tensor):
        return self.gen_aligner(self.gen_embed(image_ids))
```

配合着查看权重文件中的config，结构更加清晰。

<div align="center">
    <img src="https://github.com/user-attachments/assets/84209449-e890-4dba-9083-60f63299c62c" alt="Image 2" style="width: 50%;">
</div>

## Janus 理解文本生成代码
* 调用模型的 generate 方法生成回答。
* 输入参数包括：
  * input_ids: 文本输入的 token ID 序列。
  * inputs_embeds: 处理后的嵌入向量。
  * position_ids: 位置 ID 序列。
  * attention_mask: 注意力掩码，用于指示哪些位置是有效的输入。
  * pad_token_id, bos_token_id, eos_token_id: 分别表示填充、开始和结束的特殊 token ID。
  * max_new_tokens: 最大生成的新 token 数量，这里设置为 128。
  * do_sample: 是否使用采样生成文本，这里设置为 False，表示使用贪婪解码。
  * use_cache: 是否使用缓存机制加速生成。

```python
vl_gpt = JanusMultiModalityCausalLM.from_pretrained(args.model_path, dtype=args.dtype)
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)
image_processer = JanusImageProcessor.from_pretrained(args.model_path)
vl_chat_processor: JanusVLChatProcessor = JanusVLChatProcessor(image_processer, tokenizer)

conversation = [
    {
        "role": "User",
        "content": f"<image_placeholder>\n{args.question}",
        "images": [args.image_file],
    },
    {"role": "Assistant", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
device = prepare_inputs["pixel_values"].place
prepare_inputs.to(device, dtype=args.dtype)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
bs, seq_len = prepare_inputs.attention_mask.shape
position_ids = paddle.arange(seq_len, dtype=paddle.int64).reshape([1, -1])
outputs = vl_gpt.language_model.generate(
    input_ids=prepare_inputs["input_ids"],
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,  # 512,
    do_sample=False,
    use_cache=True,
)
answer = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
```

## Janus 图像生成代码
* 方法: generate
* 参数:
  * **mmgpt**：JanusMultiModalityCausalLM类就是一个Janus模型的实例，负责生成图像和文本。
  * **vl_chat_processor**: 多模态对话处理器，用于处理文本和图像的输入。
  * **prompt**: 输入的文本提示，用于引导图像生成。
  * **temperature**: 采样温度，控制生成的随机性。值越低，生成结果越稳定。
  * **parallel_size**: 并行生成的图像数量。
  * **cfg_weight**: Classifier-Free Guidance（CFG）权重，用于控制条件生成和无条件生成的混合比例。
  * **image_token_num_per_image**: 每张图像对应的 token 数量。
  * **img_size**: 生成图像的尺寸。
  * **patch_size**: 图像分割的 patch 尺寸。
* 步骤:
  * 文本处理：使用vl_chat_processor的分词器将文本提示编码为输入ID，然后转换为Paddle张量。
  * 初始化token：创建一个用于存储输入token和生成图像token的张量。对于并行生成的每个样本，都复制输入token，并在奇数索引的样本中插入填充token。
  * 输入Embedding：将token转换为模型可以理解的Embedding形式。
  * 生成图像token：通过一个循环，逐步生成图像的每个token。在每个步骤中：
    * 更新position id 以反映当前token生成的位置序号。
    * 使用模型的语言模型部分生成下一个token的概率分布。
    * 根据条件和无条件生成的 logits 以及温度调整概率分布。
    * 使用paddle.multinomial根据调整后的概率分布采样下一个token。
    * 使用生成的token生成图像Embedding，并更新输入Embedding以用于下一次迭代。
  * 解码图像：将生成的图像token解码为图像数据。
  * 后处理和保存：将解码后的图像数据标准化为0-255之间的整数，并保存为JPEG文件。

```python
def generate(
    mmgpt,
    vl_chat_processor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 2,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = paddle.to_tensor(data=input_ids.input_ids, dtype="int64")
    tokens = paddle.zeros(shape=(parallel_size * 2, len(input_ids)), dtype="int32")
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)  # [4, 50, 2048]
    generated_tokens = paddle.zeros(shape=(parallel_size, image_token_num_per_image), dtype="int32")
    batch_size, seq_length = inputs_embeds.shape[:2]
    for i in tqdm(range(image_token_num_per_image)):
        batch_size, seq_length = inputs_embeds.shape[:2]

        past_key_values_length = outputs.past_key_values[0][0].shape[1] if i != 0 else 0
        position_ids = paddle.arange(past_key_values_length, seq_length + past_key_values_length).expand(
            (batch_size, seq_length)
        )

        outputs = mmgpt.language_model.llama(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,  # [4, 1, 2048]
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = paddle.nn.functional.softmax(x=logits / temperature, axis=-1)
        next_token = paddle.multinomial(x=probs, num_samples=1)

        generated_tokens[:, i] = next_token.squeeze(axis=-1)
        next_token = paddle.concat(x=[next_token.unsqueeze(axis=1), next_token.unsqueeze(axis=1)], axis=1).reshape(
            [-1]
        )
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(axis=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype="int32"), shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to("float32").cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    os.makedirs("janus_generated_samples", exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join("janus_generated_samples", "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)
```

## Janus在PaddleMIX里快速体验
通过解析代码我们也更深入地理解模型的实现细节和技术创新，快跟着我们的aistudio教程一起来动手实践一下吧！
aistudio教程链接：[【PaddleMIX】快速体验DeepSeek的多模态理解生成模型 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/8798721)

我们以Janus-Pro-1B为例，在单卡V100上只需7G显存即可推理完成图像理解和图像生成。

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

安装PaddleMIX环境依赖包
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
```python
# Janus/Janus-Pro understanding
python paddlemix/examples/janus/run_understanding_inference.py \
    --model_path="deepseek-ai/Janus-Pro-1B" \
    --image_file="paddlemix/demo_images/examples_image1.jpg" \
    --question="描述一下这个图片。" \
    --dtype="bfloat16"
```
图片
<div align="left">
    <img src="../../paddlemix/demo_images/examples_image1.jpg" style="width: 50%;">
</div>
输出结果：

这张图片展示了一只红熊猫，它正趴在木板上，背景是一些树枝和绿色的树叶。红熊猫的毛色主要是棕色和白色，它的耳朵和脸部有明显的白色毛发，眼睛周围有白色的斑纹。红熊猫看起来非常可爱，它似乎在休息或观察周围的环境。

## 图像生成：
运行以下命令即可：
```python
# Janus/Janus-Pro generation
python paddlemix/examples/janus/run_generation_inference.py \
    --model_path="deepseek-ai/Janus-Pro-1B" \
    --prompt="江边有一艘船。" \
    --dtype="bfloat16"
```
<div align="left">
    <img src="https://github.com/user-attachments/assets/99dc87e6-8e23-4b5c-b449-09cee731fcdd" style="width: 50%;">
</div>

## 总结
DeepSeek 的 Janus 和 Janus-Pro 在多模态理解与生成领域展现了强大的能力。Janus 通过解耦视觉编码，为多模态任务提供了一个灵活的框架。而 Janus-Pro 则通过优化训练策略、扩展数据规模和增加模型参数，进一步提升了模型的性能。
PaddleMIX中已经复现了Janus 和 Janus-Pro 的推理流程，通过解析代码我们也更深入地理解模型的实现细节和技术创新，快跟着aistudio教程链接一起动手实践一下吧！

论文链接：

https://arxiv.org/abs/2410.13848 Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation

https://arxiv.org/pdf/2501.17811 Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling

项目地址：

https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/janus

aistudio教程链接：

[【PaddleMIX】快速体验DeepSeek的多模态理解生成模型 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/8798721)