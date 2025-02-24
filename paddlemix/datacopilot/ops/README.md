# PaddleMix 数据处理算子文档

## 目录
- [1. 转换算子](#1-转换算子)
  - [1.1 llava转换算子](#11-llava转换算子)
    - [1.1.1 llava_convert](#111-llava_convert)
- [2. 过滤算子](#2-过滤算子)
  - [2.1 基础过滤算子](#21-基础过滤算子)
    - [2.1.1 valid_data_filter](#211-valid_data_filter)
      - [2.1.1.1 image_compliance_operator](#2111-image_compliance_operator)
      - [2.1.1.2 conversation_compliance_operator](#2112-conversation_compliance_operator)
  - [2.2 文本过滤算子](#22-文本过滤算子)
    - [2.2.1 conversation_length_filter](#221-conversation_length_filter)
    - [2.2.2 average_line_length_filter](#222-average_line_length_filter)
    - [2.2.3 maximum_line_length_filter](#223-maximum_line_length_filter)
    - [2.2.4 conversation_percentage_filter](#224-conversation_percentage_filter)
    - [2.2.5 token_num_filter](#225-token_num_filter)
    - [2.2.6 alphanumeric_ratio_filter](#226-alphanumeric_ratio_filter)
    - [2.2.7 stopwords_ratio_filter](#227-stopwords_ratio_filter)
    - [2.2.8 special_characters_filter](#228-special_characters_filter)
    - [2.2.9 language_id_filter](#229-language_id_filter)
    - [2.2.10 text_action_filter](#2210-text_action_filter)
    - [2.2.11 text_entity_dependency_filter](#2211-text_entity_dependency_filter)
    - [2.2.12 char_ngram_repetition_filter](#2212-char_ngram_repetition_filter)
    - [2.2.13 word_ngram_repetition_filter](#2213-word_ngram_repetition_filter)
    - [2.2.14 conversation_hash_filter](#2214-conversation_hash_filter)
      - [2.2.14.1 simhash_duplicate_operator](#22141-simhash_duplicate_operator)
      - [2.2.14.2 minhash_duplicate_operator](#22142-minhash_duplicate_operator)
    - [2.2.15 llm_judge_filter](#2215-llm_judge_filter)
  - [2.3 图像过滤算子](#23-图像过滤算子)
    - [2.3.1 image_filesize_filter](#231-image_filesize_filter)
    - [2.3.2 image_ration_filter](#232-image_ration_filter)
    - [2.3.3 image_resolution_filter](#233-image_resolution_filter)
    - [2.3.4 image_hash_filter](#234-image_hash_filter)
  - [2.4 图文过滤算子](#24-图文过滤算子)
    - [2.4.1 image_clip_filter](#241-image_clip_filter)
- [3. 分析算子](#3-分析算子)
  - [3.1 基础分析算子](#31-基础分析算子)
    - [3.1.1 base_analysis_pipeline](#311-base_analysis_pipeline)
      - [3.1.1.1 analyze_dataset_statistics](#3111-analyze_dataset_statistics)
      - [3.1.1.2 analyze_language_distribution](#3112-analyze_language_distribution)
      - [3.1.1.3 analyze_image_paths](#3113-analyze_image_paths)
      - [3.1.1.4 analyze_data_anomalies](#3114-analyze_data_anomalies)
      - [3.1.1.5 analyze_conversation_tokens](#3115-analyze_conversation_tokens)
  - [3.2 进阶分析算子](#32-进阶分析算子)
    - [3.2.1 description_analysis](#321-description_analysis)
    - [3.2.2 quality_analysis](#322-quality_analysis)
- [4. 可视化算子](#4-可视化算子)
  - [4.1 lda可视化算子](#41-lda可视化算子)
    - [4.1.1 lda_topic_clustering](#411-lda_topic_clustering)
- [5. 生成算子](#5-生成算子)
  - [5.1 多模态生成算子](#51-多模态生成算子)
    - [5.1.1 generate_qna_for_images](#511-generate_qna_for_images)

 

## 1. 转换算子

### 1.1 llava转换算子

#### 1.1.1 llava_convert

**功能介绍**:  
将 llava 数据集转换为 paddlemix 标准格式，处理图像路径、对话配对，并过滤无效数据。

**参数说明**:
- `image_path_prefix` (str, 可选): 图像路径的前缀，用于拼接 json 中的 image 路径。

**使用示例**:
```python
dataset = MMDataset.from_json(anno_path)
dataset = dataset.llava_convert()
dataset = dataset.llava_convert(
  image_path_prefix='datasets/llava/valid_images/'
)
```

**输入输出**:
- 输入：llava 格式对话列表的原始数据集，形如：

```json
{
  "id": "000000033471",
  "image": "coco/train2017/000000033471.jpg",
  "conversations": [
      {
      "from": "human",
      "value": "<image>\nWhat are the colors of the bus in the image?"
    },
    {
      "from": "gpt",
      "value": "The bus in the image is white and red."
    }
  ]
}
```


- 输出：paddlemix 支持的对话列表的数据集，形如:

```json
{
  "image": "image_path_prefix/coco/train2017/000000033471.jpg",
  "conversations": [
    [
      "<image>\nWhat are the colors of the bus in the image?",
      "The bus in the image is white and red."
    ]
  ]
}
```

注：本文档其他所有的算子，都需要先转换格式方可使用。

---

## 2. 过滤算子

- 过滤算子中的输入和输出都是包含 image 和 conversations（paddlemix 支持的对话列表）的数据集。
- 所有算子都可使用默认参数直接调用。
- 调用示例：`dataset = dataset.filter_name()`。
- 训练前建议先使用2.1的基础过滤算子进行异常数据清洗，防止训练过程中报错。

### 2.1 基础过滤算子

#### 2.1.1 valid_data_filter

`valid_data_filter()` 下联合了 `image_compliance_operator` 以及 `conversation_compliance_operator`算子，分别用于图像和文本的无效数据过滤，通过调用 `valid_data_filter()` 同时过滤这两种模态的异常数据。

**使用示例**:
```python
dataset = dataset.valid_data_filter()
```

##### 2.1.1.1 image_compliance_operator

**功能介绍**:  
过滤数据集中无效的图像数据，确保数据符合使用要求。

**功能详情**:
- 图像有效性检查：
  - 验证图片文件存在且可加载。

##### 2.1.1.2 conversation_compliance_operator

**功能介绍**:  
过滤数据集中无效的对话数据，确保数据符合使用要求。

**功能详情**:
- 对话合规性检查：
  - 确保 conversations 为有效的列表结构。
  - 每条对话必须是 [human_message, gpt_message] 格式。
  - 对话内容不能包含 USER 或 ASSISTANT 关键字。
  - 确保对话内容为非空字符串。

---

### 2.2 文本过滤算子

#### 2.2.1 conversation_length_filter

**功能介绍**:  
过滤数据集中会话内容过长的条目。

**功能详情**:
- 将 conversations 中的所有内容拼接成一个字符串。
- 去除 <image> 占位符及其换行符。
- 检查拼接后的字符串长度是否小于 max_length。

**参数说明**:
- `max_length` (int, 默认值: 2048): 会话的最大字符长度。

**使用示例**:
```python
dataset = dataset.conversation_length_filter(
  max_length=2048
)
```

---

#### 2.2.2 average_line_length_filter

**功能介绍**:  
根据会话的平均行长度过滤数据集中的样本。

**参数说明**:
- `min_length` (int, 默认值: 10): 每行文本的最小平均长度。
- `max_length` (float, 默认值: float('inf')): 每行文本的最大平均长度。

**使用示例**:
```python
dataset = dataset.average_line_length_filter(
    min_length=15,  
    max_length=50  
)
```

---

#### 2.2.3 maximum_line_length_filter

**功能介绍**:  
根据会话的最大行长度过滤数据集中的样本。

**参数说明**:
- `min_length` (int, 默认值: 10): 最大行长度的最小值。
- `max_length` (float, 默认值: float('inf')): 最大行长度的最大值。

**使用示例**:
```python
dataset = dataset.maximum_line_length_filter(
    min_length=10, 
    max_length=128  
)
```

---

#### 2.2.4 conversation_percentage_filter 

**功能介绍**:  
根据对话数量的百分位数范围，过滤数据集中对话数量过少或过多的条目。

**参数说明**:
- `min_percentile` (float, 默认值: 5): 最小百分位数，保留对话数量大于等于该百分位数的条目。
- `max_percentile` (float, 默认值: 95): 最大百分位数，保留对话数量小于等于该百分位数的条目。

**使用示例**:
```python
dataset = dataset.conversation_percentage_filter(
    min_percentile=5, 
    max_percentile=95 
)
```

---

#### 2.2.5 token_num_filter

**功能介绍**:  
用于根据会话的 token 数量过滤数据集。

**功能详情**:
- 加载指定的 tokenizer 模型。
- 统计会话的 token 数量。
- 如果样本的 token 数量小于 min_tokens 或大于 max_tokens，则过滤掉该样本。

**参数说明**:
- `tokenizer_model` (str, 默认值: "Qwen/Qwen2.5-7B"): 使用的 tokenizer 模型名称。
- `min_tokens` (int, 默认值: 10): 样本的最小 token 数量。
- `max_tokens` (int, 默认值: sys.maxsize): 样本的最大 token 数量。

**使用示例**:
```python
dataset = dataset.token_num_filter(
    tokenizer_model="Qwen/Qwen2.5-7B",  
    min_tokens=10,                   
    max_tokens=512                      
)
```

---

#### 2.2.6 alphanumeric_ratio_filter

**功能介绍**:  
根据文本中字母或数字字符占总字符数的比例过滤数据集中的样本。

**参数说明**:
- `min_ratio` (float, 默认值: 0.25): 样本中文本的最小字母或数字字符比例。
- `max_ratio` (float, 默认值: float('inf')): 样本中文本的最大字母或数字字符比例。

**使用示例**:
```python
dataset = dataset.alphanumeric_ratio_filter(
    min_ratio=0.25,  
    max_ratio=0.75  
)
```

---

#### 2.2.7 stopwords_ratio_filter

**功能介绍**:  
根据样本中的停用词比例对数据集进行过滤，通过设置最小停用词比例，筛选出停用词比例大于或等于指定值的样本。

**功能详情**:
- 使用 NLTK 的 stopwords 资源获取英语停用词列表。
- 对样本中的问答对内容进行分词，统计停用词的数量。
- 如果停用词比例低于 min_ratio，则过滤掉该样本。

**参数说明**:
- `min_ratio` (float, 默认值: 0.25): 样本中停用词比例的最小值。停用词比例低于该值的样本将被过滤。

**使用示例**:
```python
dataset = dataset.stopwords_ratio_filter(
    min_ratio=0.25 
)
```

---

#### 2.2.8 special_characters_filter

**功能介绍**:  
通过计算每个样本中特殊字符占总字符的比例，并根据指定的上下限范围筛选出符合条件的样本。

**功能详情**:
- 统计文本中指定的特殊字符（如 |, @, # 等）的数量。
- 计算特殊字符占总字符的比例：特殊字符数量 / 总字符数量。
- 根据提供的 min_ratio 和 max_ratio，筛选出特殊字符比例在指定范围内的样本。

**参数说明**:
- `min_ratio` (float, 默认值: 0.0): 特殊字符比例的最小值（下限）。
- `max_ratio` (float, 默认值: 0.25): 特殊字符比例的最大值（上限）。

**使用示例**:
```python
dataset = dataset.special_characters_filter(
    min_ratio=0.25,  
    max_ratio=0.75  
)
```

---

#### 2.2.9 language_id_filter

**功能介绍**:  
通过 FastText 模型检测每个样本的语言，并根据用户指定的语言代码和最小置信度阈值筛选出符合条件的样本。

**功能详情**:
- 如果本地未找到指定的 FastText 模型文件，会自动从官方链接下载模型并加载。
- 检查检测到的语言是否在用户指定的语言列表中（lang）。
- 检查语言置信度分数是否高于用户提供的最小阈值（min_score）。
- 仅保留符合语言要求和置信度阈值的样本。

**参数说明**:
- `lang` (Union[str, List[str], None], 默认值: None): 允许的语言代码，可以是单个语言代码（如 "en"）、语言代码列表（如 ["en", "zh"]），或 None（不限制语言）。
- `min_score` (float, 默认值: 0.8): 最小语言置信度分数，用于确保语言检测结果可靠。


**使用示例**:
```python
dataset = dataset.language_id_filter(
    lang=["en", "fr"],  
    min_score=0.9     
)
```

---


#### 2.2.10 text_action_filter

**功能介绍**:  
通过检测样本中的动词数量，根据指定的最小动词数量过滤数据集。使用 spaCy 模型进行语言处理，支持基于英语的动词检测规则。

**功能详情**:
- 使用 spaCy 模型处理样本中的文本内容，提取动词。
- 通过 pos_ 和 tag_ 属性检测动词，例如 VERB 表示动词的词性。
- 如果样本中的动词数量小于 min_action_num，则过滤掉该样本。

**参数说明**:
- `lang` (str, 默认值: 'en'): 文本的语言，当前支持 'en'（英语）。
- `min_action_num` (int, 默认值: 1): 样本中动词的最小数量，动词数量小于该值的样本将被过滤。

**使用示例**:
```python
dataset = dataset.text_action_filter(
    lang="en",  
    min_action_num=2  
)
```

---

#### 2.2.11 text_entity_dependency_filter

**功能介绍**:  
通过检测样本中的实体依赖关系，根据每个实体的依赖边数量对数据集进行过滤。用户可以选择不同的筛选策略（any 或 all），并设置依赖边的最小数量。

**功能详情**:  
- 使用 spaCy 模型处理样本中的文本内容。
- 通过 POS 和 Tag 的规则识别实体，例如名词、专有名词和代词。
- 统计每个实体的依赖边数量，包括实体本身的依赖关系和其他词对实体的依赖关系。
- 根据用户指定的策略（any 或 all）以及最小依赖边数量筛选样本。

**参数说明**:
- `lang` (str, 默认值: 'en'): 文本的语言。当前支持 'en'（英语）。
- `min_dependency_num` (int, 默认值: 1): 每个实体的最小依赖边数量。
- `any_or_all` (str, 默认值: 'any'): 筛选策略，可选值为 'any'（只要有一个实体满足条件）或 'all'（所有实体都必须满足条件）。

**使用示例**:
```python
dataset = dataset.text_entity_dependency_filter(
    lang="en",              
    min_dependency_num=2,   
    any_or_all="any"      
)
```

---

### 2.2.12 char_ngram_repetition_filter

**功能介绍**:
根据会话中字符 n-gram 的重复比例过滤数据集。

**功能详情**:
- 将样本中的文本按字符级别切分，并生成长度为 rep_len 的字符 n-gram。
- 统计每个 n-gram 的出现频率。
- 计算重复 n-gram 的比例，即 n-gram 出现频率大于 1 的 n-gram 数量占总 n-gram 数量的比例。
- 如果样本中的字符 n-gram 重复比例不在 [min_ratio, max_ratio] 范围内，则过滤掉该样本。

**参数说明**:
- `rep_len` (int, 默认值: 10): n-gram 的长度。
- `min_ratio` (float, 默认值: 0.0): 最小重复比例。
- `max_ratio` (float, 默认值: 0.5): 最大重复比例。

使用示例:
```python
dataset = dataset.char_ngram_repetition_filter(
    rep_len=10, 
    min_ratio=0.1, 
    max_ratio=0.4
)
```

---

#### 2.2.13 word_ngram_repetition_filter

**功能介绍**:  
通过计算样本中的词 n-gram 重复比例，根据指定的重复比例范围过滤数据集。

**功能详情**:  
- 将样本中的文本按空格分词，并生成长度为 rep_len 的词 n-gram。
- 统计每个 n-gram 的出现频率。
- 计算重复 n-gram 的比例。
- 如果样本中的 n-gram 重复比例不在 [min_ratio, max_ratio] 范围内，则过滤掉该样本。

**参数说明**:
- `rep_len` (int, 默认值: 10): n-gram 的长度。
- `min_ratio` (float, 默认值: 0.0): 最小重复比例。
- `max_ratio` (float, 默认值: 0.5): 最大重复比例。

**使用示例**:
```python
dataset = dataset.word_ngram_repetition_filter(
    rep_len=10, 
    min_ratio=0.1, 
    max_ratio=0.4
)
```

---

#### 2.2.14 conversation_hash_filter

**功能介绍**:  
`conversation_hash_filter` 是一个用于去除重复问答对的算子，它统一调用 `simhash_duplicate_operator` 或 `minhash_duplicate_operator` 算子进行处理。

**参数说明**:
- `method` (str, 默认值: "simhash"): 去重方法。
  - "simhash": 使用 SimHash 算法（基于汉明距离）检测文本重复。
  - "minhash": 使用 MinHashLSH 算法（基于 Jaccard 相似度）检测文本重复。
- `threshold` (float, 默认值: 0.8): 相似度阈值，影响去重的严格程度：
  - SimHash 汉明距离的比例，1 - threshold 是允许的最大汉明距离。例如，threshold=0.8 表示允许最多 20% 的汉明距离。
- `num_perm` (int, 默认值: 128):
  - MinHash 的置换次数，仅在 method="minhash" 时有效。值越大，MinHash 签名的精度越高，但计算开销也会增加。


**使用示例**:
```python
dataset = dataset.conversation_hash_filter(
    method="simhash",
    threshold=0.8,
)
```

#### 2.2.14.1 simhash_duplicate_operator

**功能介绍**:  
使用simhash算法对问答对进行去重。

**功能详情**:
- 生成 SimHash 指纹: 将输入文本（问题和答案拼接后的内容）生成一个 64 位的 SimHash 指纹。
- 计算汉明距离: 遍历已记录的 SimHash 指纹集合，计算当前文本的 SimHash 指纹与集合中每个指纹的汉明距离。汉明距离表示两个二进制指纹之间的位差，值越小表示文本越相似。
- 判定是否重复: 比较汉明距离是否小于等于允许的最大汉明距离（(1 - threshold) * 64）。如果小于或等于，则认为文本重复；否则，记录该 SimHash 指纹。
- 更新已记录指纹集合: 如果当前文本不重复，将其 SimHash 指纹加入已记录集合。

#### 2.2.14.2 minhash_duplicate_operator

**功能介绍**:  
使用minhash算法对问答对进行去重。

**功能详情**:
- 生成 MinHash 签名: 将输入文本（问题和答案拼接后的内容）分词，并对每个词生成 MinHash 签名。MinHash 签名由一组哈希值组成，数量由参数 num_perm 决定（哈希次数越多，准确性越高）。
- 查询相似文本: 使用 MinHashLSH查询已存储的 MinHash 签名集合，检测当前文本是否与集合中的已有文本相似。Jaccard 相似度是通过 MinHash 签名的交集和并集估算得出的。
- 判定是否重复: 如果查询结果中存在相似文本（Jaccard 相似度 ≥ threshold），则认为文本重复。否则，将当前文本的 MinHash 签名存入 MinHashLSH。
- 更新 MinHashLSH: 如果当前文本不重复，为其生成唯一键，并将其 MinHash 签名插入到 MinHashLSH 中。

---

#### 2.2.15 llm_judge_filter

**功能介绍**:  
利用 LLM 模型分析数据集中的问答对，根据模型的评分过滤掉质量较差的问答对。

**参数说明**:
- `model_name` (str, 默认值: "Qwen/Qwen2.5-7B"): 使用的 LLM 模型名称。
- `batch_size` (int, 默认值: 1): 每次处理的问答对数量。

**功能详情**:
- 使用指定的 LLM 模型对每个问答对生成评价并提取评分。
- 评分范围为 1 到 4，保留评分大于等于 3 的问答对。

**使用示例**:
```python
dataset = dataset.llm_judge_filter(
    model_name="Qwen/Qwen2.5-7B",
    batch_size=1
)
```

---

### 2.3 图像过滤算子

#### 2.3.1 image_filesize_filter

**功能介绍**:  
过滤数据集中图像文件大小不符合要求的样本。

**参数说明**:
- `min_size_kb` (float, 默认值: 10): 图像文件的最小大小（KB）。
- `max_size_kb` (float, 默认值: None): 图像文件的最大大小（KB）。

**使用示例**:
```python
dataset = dataset.image_filesize_filter(
    min_size_kb=10,
    max_size_kb=1024
)
```

---

#### 2.3.2 image_ration_filter

**功能介绍**:  
过滤数据集中宽高比不符合指定范围的图像。

**参数说明**:
- `min_ratio` (float, 默认值: 0.333): 图像的最小宽高比。
- `max_ratio` (float, 默认值: 3.0): 图像的最大宽高比。

**使用示例**:
```python
dataset = dataset.image_ration_filter(
    min_ratio=0.333,
    max_ratio=3.0
)
```

---

#### 2.3.3 image_resolution_filter

**功能介绍**:  
过滤数据集中分辨率不符合指定范围的图像样本。

**参数说明**:
- `min_width` (float, 默认值: 112): 图像的最小宽度。
- `min_height` (float, 默认值: 112): 图像的最小高度。
- `max_width` (float, 默认值: None): 图像的最大宽度。
- `max_height` (float, 默认值: None): 图像的最大高度。

**使用示例**:
```python
dataset = dataset.image_resolution_filter(
    min_width=112,
    min_height=112,
    max_width=1920,
    max_height=1080
)
```

---

#### 2.3.4 image_hash_filter

**功能介绍**:  
通过图像哈希值对数据集中的图像去重。

**参数说明**:
- `hash_method` (str, 默认值: "phash"): 图像哈希算法类型，支持 "phash"、"dhash" 和 "average_hash"

**功能详情**:
- 支持多种哈希算法：
  - phash: 感知哈希，适用于检测图像内容相似性。
  - dhash: 差异哈希，计算图像梯度变化。
  - average_hash: 平均哈希，基于图像平均像素值计算。

**使用示例**:
```python
dataset = dataset.image_hash_filter(
  hash_method="phash"
)
```

---

### 2.4 图文过滤算子

#### 2.4.1 image_clip_filter

**功能介绍**:  
使用 CLIP 模型对数据集中的问答对进行过滤，根据图像与文本的相似度移除低置信度的问答对。

**功能详情**:
- 文本预处理：对问答对进行清理，移除占位符和多余换行符。
- 自动跳过包含坐标形式的问答对。
- 使用 CLIP 模型计算图像-问答对相似度。
- 可选保存低置信度样本图像。

**参数说明**:
- `model_name` (str, 默认值: "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"): 使用的 CLIP 模型名称。
- `threshold` (float, 默认值: 0.25): 图像与文本相似度的置信度阈值。
- `batch_size` (int, 默认值: 8): 批量处理的问答对数量。
- `save_images` (bool, 默认值: False): 是否保存低置信度的样本图像。
- `save_dir` (str, 默认值: "./low_confidence_images"): 保存低置信度图像的目录。


**使用示例**:
```python
config = CLIPFilterConfig(
    threshold=0.3,
    batch_size=8,
    save_images=True,
    save_dir="./filtered_images"
)
dataset = dataset.image_clip_filter(config=config)
```

---


## 3. 分析算子

### 3.1 基础分析算子

#### 3.1.1 base_analysis_pipeline

**功能介绍**:
- 调用多个分析算子（如数据统计、语言分布、图片路径验证等）对数据集进行全面分析。
- 根据传入的 `analysis_flags` 控制运行哪些分析任务。
- 将分析结果保存到指定的输出目录，并可视化分析结果。

**功能详情**:
- **数据统计分析**: 统计数据集的总记录数、有效记录数、对话统计等。
- **语言分布分析**: 分析人类消息和助手消息的语言分布，检测语言不匹配的情况。
- **图片路径验证**: 验证图片路径的分布和存在性，统计缺失图片数量。
- **数据异常检测**: 检测缺失字段或对话内容为空的记录。
- **Token 分析**: 分析对话内容的 Token 分布，统计高频和低频 Token。

**参数说明**:
- analysis_flags (Dict[str, bool]): 控制各个分析任务的布尔值字典，默认值如下：
  - "analyze_dataset": True（是否运行数据统计分析）
  - "analyze_languages": True（是否运行语言分布分析）
  - "analyze_image_paths": True（是否运行图片路径验证）
  - "analyze_anomalies": True（是否运行数据异常检测）
  - "analyze_tokens": True（是否运行 Token 分析）
- output_dir (str): 保存分析结果的目录路径，默认值为 "output_directory"。

**输入输出**:

- 输入:
  - dataset (MMDataset): 待分析的多模态数据集。

- 输出: Dict[str, Any]: 包含所有分析任务结果的字典，每个任务的结果对应一个键：
  - "dataset_statistics": 数据统计分析结果。
  - "language_distribution": 语言分布分析结果。
  - "image_path_validation": 图片路径验证结果。
  - "anomaly_detection": 数据异常检测结果。
  - "token_analysis": Token 分析结果。

**使用示例**:
```python
analysis_flags = {
    "dataset_statistics": True,
    "language_distribution": True,
    "image_path_analysis": True,
    "data_anomalies": True,
    "conversation_tokens": False
}
results = dataset.base_analysis_pipeline(analysis_flags=analysis_flags, output_dir="analysis_results")
```


##### 3.1.1.1 analyze_dataset_statistics

**功能介绍**:
- 分析数据集的基本统计信息，包括总记录数、唯一图片数、对话统计等。
- 统计有效和无效记录。

**输入输出**:
- 输入: 
  - dataset (MMDataset): 待分析的多模态数据集。

- 输出: Dict[str, Any]: 包含数据集统计信息的字典，包括：
  - total_records: 数据集中的总记录数。
  - unique_images: 数据集中唯一图片的数量。
  - total_conversations: Q&A 对的总数。
  - max_conversations: 单条记录中最多的 Q&A 对数量。
  - min_conversations: 单条记录中最少的 Q&A 对数量。
  - avg_conversations: 平均 Q&A 对数量。
  - invalid_item_count: 无效记录的数量。
  - valid_items: 有效记录的列表。


##### 3.1.1.2 analyze_language_distribution

**功能介绍**:
- 检测对话中人类消息和助手消息的语言分布。
- 统计人类消息和助手消息的总数。
- 检测人类和助手语言不匹配的情况。

**输入输出**:
- 输入:
  - dataset (MMDataset): 待分析的多模态数据集。
  - lang_model (fasttext.FastText._FastText): 已加载的 FastText 语言检测模型。

- 输出: Dict[str, Any]: 语言分布统计信息，包括以下字段：
  - "human_message_count": 人类消息的总数量。
  - "assistant_message_count": 助手消息的总数量。
  - "mismatched_language_pairs_count": 人类和助手语言不匹配的对数。
  - "languages_distribution": 不同语言的分布情况（字典）。


##### 3.1.1.3 analyze_image_paths

**功能介绍**:
- 验证数据集中图片路径的分布和存在性。
- 统计总图片数量、缺失图片数量和路径分布。

**输入输出**:
- 输入:
  - dataset (MMDataset): 待分析的多模态数据集。

- 输出: Dict[str, Any]: 图片路径验证结果，包括以下字段：
  - "total_images": 图片路径的总数量。
  - "missing_images": 缺失的图片数量（路径不存在的图片）。
  - "path_distribution": 图片路径的分布情况（按路径目录统计）。


##### 3.1.1.4 analyze_data_anomalies

**功能介绍**:
- 检测数据集中缺少必要字段（如图片路径或对话内容）的记录。
- 检测对话内容为空的记录。
- 保存异常记录到指定目录的 JSON 文件中。

**输入输出**:
- 输入:
  - dataset (MMDataset): 待分析的多模态数据集。
  - output_dir (str): 保存异常记录的目录路径。

- 输出: Dict[str, int]: 异常统计信息，包括以下字段：
  - "missing_field_count": 缺失必要字段的记录数。
  - "empty_conversation_count": 对话内容为空的记录数。


##### 3.1.1.5 analyze_conversation_tokens

**功能介绍**:
- 对数据集中的对话内容进行分词分析。
- 分析人类消息和助手消息的 Token 分布。
- 统计高频和低频 Token。

**输入输出**:
- 输入:
  - dataset (MMDataset): 待分析的多模态数据集。
  - tokenizer (AutoTokenizer): 用于分词的 PaddleNLP Tokenizer 实例。

- 输出: Dict[str, Any]: Token 分析结果，包括以下字段：
  - "human":
    - "total_tokens": 人类消息的总 Token 数量。
    - "high_freq_tokens": 人类消息中高频 Token 的统计（字典）。
    - "low_freq_tokens": 人类消息中低频 Token 的统计（字典）。
  - "assistant":
    - "total_tokens": 助手消息的总 Token 数量。
    - "high_freq_tokens": 助手消息中高频 Token 的统计（字典）。
    - "low_freq_tokens": 助手消息中低频 Token 的统计（字典）。

---

### 3.2 进阶分析算子

#### 3.2.1 description_analysis

**功能介绍**:
对多轮对话进行解析、属性提取、结果过滤和数据清洗，支持从图像相关对话中提取结构化信息，如颜色、形状、位置、大小、方向、关系、动作等。

**功能详情**:
- 自动加载指定的预训练模型和分词器，支持高效的批量推理。
- 使用预定义的 prompt，指导模型从对话内容中提取关键信息。包括：提取颜色、形状等具体属性；抽取对象之间的关系、动作等动态信息。
- 对未提及或缺失的属性，自动填充为 "N/A" 或默认值，保证输出 JSON 数据结构完整。
- 对提取的属性进行频次统计，忽略无效数据（如 "N/A"），并生成每个属性类别的高频信息。
- 过滤无效的解析结果，确保只有成功提取的对话与图像相关联，避免重复或无效记录干扰分析结果。

**输入输出**:
- 输入:
  - dataset (MMDataset): 待分析的多模态数据集。
  - model_name (str, 默认值: "Qwen/Qwen2.5-7B"): 使用的PaddleNLP预训练语言模型名称.
  - batch_size (int, 默认值: 1): 模型推理时的批量大小。

- 输出:
  - dataset (MMDataset): 属性分析过滤后的多模态数据集。

**使用示例**:
```python
dataset = dataset.description_analysis(
  model_name= "Qwen/Qwen2.5-7B", 
  batch_size=1
)
```

---

#### 3.2.2 quality_analysis

**功能介绍**:
对多轮对话的图像描述质量进行评估，基于预定义的多种评估指标（如图文匹配、对象细节描述等），提供结构化的评价结果，支持灵活选择评估维度。

**功能详情**:
- 自动加载预定义的高性能预训练模型（如 Qwen2.5-7B）和分词器，支持高效的批量推理。
- 使用多种预定义的评估标准（如图文匹配、对象细节描述、文本质量等），根据用户选择的分析标志 (`analysis_flags`) 对图像描述进行多维度评分。
- 支持灵活的分析维度配置，用户可以选择启用或禁用特定的评估指标。
- 根据每个评估标准生成详细的评分和解释，结果以结构化的 JSON 格式输出。
- 提供对多样化对话生成的质量评估，适配不同的任务需求。

支持的评估维度:
- 图文匹配（image_text_matching）：评估图像和文本描述的匹配程度。
- 对象细节描述（object_detail_fulfillment）：评估文本对图像中对象细节的描述程度。
- 文本质量（caption_text_quality）：评估文本描述的语法正确性、流畅性和多样性。
- 语义信息理解（semantic_understanding）：评估文本是否提供了额外的语义信息。


**输入输出**:
- 输入:
  - dataset (MMDataset): 包含图像路径和对话内容的多模态数据集。
  - model_name (str): 使用的预训练语言模型名称（默认值: "Qwen/Qwen2-VL-7B-Instruct"）。
  - quality_analysis_flags (Dict[str, bool], 可选): 用于控制评估维度的标志字典。

- 输出: Dict[str, Any]: 包含每张图像的评估结果。

**使用示例**:
```python
quality_analysis_flags = {
    "image_text_matching": True,
    "object_detail_fulfillment": False,
    "caption_text_quality": False,
    "semantic_understanding": False,
}
dataset_results = dataset.quality_analysis(
    model_name="Qwen/Qwen2-VL-7B-Instruct", 
    quality_analysis_flags=quality_analysis_flags  
)
```

---


## 4. 可视化算子

### 4.1 lda可视化算子

#### 4.1.1 lda_topic_clustering

**功能介绍**:
对多轮对话中的文本数据进行主题聚类分析，结合 LDA 和 T-SNE 进行降维与可视化，帮助揭示对话数据中的潜在主题分布以及文档之间的聚类模式。

**功能详情**:
- 文本提取: 从数据集中提取对话中的问题与回答文本，组合为单条记录供主题分析使用。
- 文本向量化: 使用 CountVectorizer 将提取的文本转换为词频矩阵，以便 LDA 模型进行主题建模。
- LDA 主题建模: 基于词频矩阵，利用 LDA 分析文档中的主题分布，并为每个文档分配主题概率分布。
- T-SNE 降维与可视化: 将 LDA 的主题分布结果通过 T-SNE 降维至二维，以便进行可视化。生成的散点图直观展示了文档在主题空间中的分布情况。
- 结果输出: 返回每个文档的主题分布、T-SNE 降维结果以及每个文档最可能的主题类别，同时保存生成的 T-SNE 可视化图像。

**输入输出**:
- 输入:dataset (MMDataset): 包含对话数据的多模态数据集。
  - num_topics (int, 默认值: 5): LDA 模型中要提取的主题数量。
  - tsne_perplexity (int, 默认值: 30): T-SNE 的 perplexity 参数，用于控制降维时的邻域范围。
  - tsne_learning_rate (int, 默认值: 200): T-SNE 的学习率。
  - tsne_n_iter (int, 默认值: 1000): T-SNE 的迭代次数。
  - random_state (int, 默认值: 42): 随机种子，确保结果可复现。
  - output_plot (str, 默认值: "lda_tsne_plot.png"): 保存 T-SNE 可视化结果的图像路径。

- 输出: 返回一个包含下列键值的字典:
  - lda_result: 每个文档的主题分布矩阵，形状为 (文档数量, num_topics)。
  - tsne_result: 每个文档的二维 T-SNE 投影结果，形状为 (文档数量, 2)。
  - topics: 每个文档最可能的主题类别（基于主题概率分布的 argmax）。
  - 同时将 T-SNE 可视化图像保存到指定路径。

**使用示例**:
```python
results = lda_topic_clustering(
    dataset=dataset,
    num_topics=5,                
    tsne_perplexity=30,          
    tsne_learning_rate=200,      
    tsne_n_iter=1000,            
    random_state=42,          
    output_plot="lda_tsne_plot.png" 
)
```

---

## 5. 生成算子

### 5.1 多模态生成算子

#### 5.1.1 generate_qna_for_images

**功能介绍**:
通过使用视觉语言模型（如 Qwen2-VL-7B-Instruct），为给定文件夹中的每张图像生成三个详细的问答对（Q&A pairs）。问答对涵盖基本视觉内容、对象关系或计数，以及背景知识或事件的推测，具体问题类型和答案内容均由模型根据图像内容生成。

**功能详情**:
- 使用预训练的视觉语言模型（如 Qwen/Qwen2-VL-7B-Instruct）处理每张图像。
- 自动生成三个问题和对应答案，问题类型涵盖：
  - 图像的基本视觉内容。
  - 图像中物体的关系或数量。
  - 图像的背景知识或事件。
- 确保生成的问题多样化且回答详细具体。

**输入输出**:
- 输入:
  - image_folder_path (str): 包含待处理图像的文件夹路径。
  - model_name (str, 默认值: "Qwen/Qwen2-VL-7B-Instruct"): 使用的视觉语言模型名称。

- 输出: 返回一个 MMDataset 对象，包含以下字段：
  - image (str): 图像的文件路径。
  - conversations (list): 图像生成的问答对列表，每个问答对以 [question, answer] 的形式存储。

**使用示例**:
```python
dataset = generate_qna_for_images(
  image_folder_path="paddlemix/demo_images", 
  model_name="Qwen/Qwen2-VL-7B-Instruct"
)
```