
# DataCopilot

<details>
<summary>Fig</summary>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c1acd673-8e2d-421d-8703-cc55ef259c48" width=500>
</div>

</details>

# DataCopilot 使用教程

## 一、简介
**DataCopilot** 是 **PaddleMIX** 提供的多模态数据处理工具箱，旨在帮助开发者高效地进行数据预处理、增强和转换等操作。通过 **DataCopilot**，你可以以低代码量的方式实现数据的基本操作，从而加速模型训练和推理的过程。

## 二、定位
DataCopilot是PaddleMIX 2.0版本新推出的多模态数据处理工具箱，理念是把数据作为多模态算法的一部分参与迭代的全流程，让开发者根据特定任务以低代码量实现数据的基本操作。

## 三、安装与导入
首先，确保你已经安装了 **PaddleMIX**。如果尚未安装，请参考 **PaddleMIX** 的官方文档进行安装。
安装完成后，你可以通过以下方式导入 **DataCopilot**：
```python
from paddlemix.datacopilot.core import MMDataset, SCHEMA
import paddlemix.datacopilot.ops as ops
```

## 四、核心概念 
工具核心概念包括Schema和Dataset。Schema用于定义多模态数据组织结构和字段名字。MMDataset作为数据操作的核心类，为存储，查看，转换，生成等操作的基本对象。

### SCHEMA
schema用于定义多模态数据格式（比如json文件里组织结构和字段名字），用于不同格式的转换，简化ops操作的逻辑，内置MM类型。
添加schema步奏，1. 在SCHEMA枚举类型增量添加标识字段（注意：不可与之前的名字重复）2. convert_schema添加和MM相互转换的逻辑。
```
class SCHEMA(Enum):
    MM = 1
```


### DATASET

核心类MMDeteset，为存储，查看，转换，生成等操作的基本对象。支持基本的运算（切片，加法，遍历等操作）。支持json数据源。内置map，filter函数，用于高效处理数据，支持多进程和多线程并发功能。支持链式调用，方便组合多种原子操作以实现复杂的功能。通过以map函数为接口实现对数据集每个元素的处理，通过register注册机制可灵活新增作用于整个数据集的通用操作功能。
```
'from_auto',
'from_json',
'export_json',
'items',
'schema',
'map',
'filter',
'shuffle'
'nonempty',
'sanitize',
'head',
'info',

# base
dataset1 = dataset[:1000]
dataset2 = dataset[-100:]

newdataset = dataset1 + dataset2 # new instance
dataset1 += dataset2 # modified dataset1 inplace

sample_item1 = dataset.shuffle()[0]
sample_item2 = dataset[random.randint(0, len(dataset))]

dataset.info() # common analysis info
dataset.head() # same as bash `head`
```


### Ops
ops包含预设的基本操作，从使用的角度分为两大类，一类是对item操作的函数，可配合map函数使用，例如质量评估，语种分析，模板扩充等；另一类是对数据集操作的函数，可单独使用，也可配合register注册机制使用。
从功能的角度分为数据分析，数据转换，数据生成，数据过滤等。
```
@register()
def info(dataset: MMDataset) -> None: ...

```

## 五、基本操作
### 1. 加载数据
使用 `MMDataset.from_json` 方法从 JSON 文件中加载数据：
```python
dataset = MMDataset.from_json('path/to/your/dataset.json')
```

使用 `MMDataset.load_jsonl` 方法从 JSONL 文件中加载数据：
```python
dataset = MMDataset.load_jsonl('path/to/your/dataset.jsonl')
```

使用 `MMDataset.from_h5` 方法从 h5 文件中加载数据：
```python
dataset = MMDataset.from_h5('path/to/your/dataset.h5')
```

### 2. 查看数据
使用 info 和 head 方法查看数据集的基本信息和前几个样本：
```python
dataset.info()
dataset.head()
```

### 3. 数据切片
支持对数据集进行切片操作，返回一个新的 MMDataset 对象：
```python
subset = dataset[:100]  # 获取前100个样本
```

### 4. 数据增强
使用 map 方法对数据集中的样本进行增强操作：
```python
def augment_data(item):
    # 定义你的数据增强逻辑
    pass

augmented_dataset = dataset.map(augment_data, max_workers=8, progress=True)
```

### 5. 数据过滤
使用 filter 方法根据条件过滤数据集中的样本：
```python
def is_valid_sample(item):
    # 定义你的过滤条件
    return True or False

filtered_dataset = dataset.filter(is_valid_sample).nonempty()  # 返回过滤后的非空数据集
```

### 6. 导出数据
使用 export_json 方法将处理后的数据集导出为 JSON 文件：
```python
augmented_dataset.export_json('path/to/your/output_dataset.json')
```

使用 export_jsonl 方法将处理后的数据集导出为 JSONL 文件：
```python
augmented_dataset.export_jsonl('path/to/your/output_dataset.jsonl')
```

使用 export_h5 方法将处理后的数据集导出为 h5 文件：
```python
augmented_dataset.export_h5('path/to/your/output_dataset.h5')
```
## 六、高级操作
### 1. 自定义 Schema
通过定义 SCHEMA 来指定数据集的字段和类型：
```python
schema = SCHEMA(
    image={'type': 'image', 'required': True},
    text={'type': 'str', 'required': True},
    label={'type': 'int', 'required': False}
)
```
使用自定义 schema 加载数据
```python
custom_dataset = MMDataset.from_json('path/to/your/dataset.json', schema=schema)
```

### 2. 批量处理
使用 batch 方法将数据集中的样本按批次处理，适用于需要批量操作的情况：
```python
batch_size = 32
batched_dataset = dataset[i: i + batch_size]
```

### 3. 数据采样
使用 shuffle 方法打乱数据集，或使用 sample 方法随机抽取样本：
```python
shuffled_dataset = dataset.shuffle()
sampled_dataset = dataset.sample(10)  # 随机抽取10个样本
```

###  4. 链式调用
```python
 MMDataset.from_json(orig_path)
        .sanitize(schema=SCHEMA.MM, max_workers=8, progress=True, )
        .map(functools.partial(ops.convert_schema, out_schema=SCHEMA.MIX), max_workers=8)
        .filter(filter_text_token, max_workers=8, progress=True, order=True)
        .nonempty()
        .export_json(new_path)
```

## 七、使用案例
1. 导入导出  
```
import functools
from paddlemix.datacopilot.core import MMDataset, SCHEMA
import paddlemix.datacopilot.ops as ops

dataset = MMDataset.from_json('./path/to/your/json/file')
print(len(dataset))

dataset.export_json('/path/to/your/output/file.json')
```

2. 字段处理  
```
# custom code 
def update_url(item: T) -> T: ...

def augment_prompt(item: T) -> T: ...

def is_wanted(item: T) -> bool: ...

# map
dataset = dataset.map(update_url, max_workers=8, progress=True)
dataset = dataset.map(augment_prompt, max_workers=8, progress=True)

# op chain
dataset = dataset.map(update_url).map(augment_prompt)

# filter
dataset = dataset.filter(is_wanted).nonempty()
```

3. LLaVA-SFT训练  
数据准备和训练流程参考项目[pp_cap_instruct](https://aistudio.baidu.com/projectdetail/7917712)

## 八、总结
**DataCopilot** 是 **PaddleMIX** 提供的一个强大且灵活的多模态数据处理工具箱。
通过掌握其基本操作和高级功能，你可以高效地处理、增强和转换多模态数据，为后续的模型训练和推理提供有力支持。