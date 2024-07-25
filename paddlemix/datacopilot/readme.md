
# DataCopilot

<details>
<summary>Fig</summary>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c1acd673-8e2d-421d-8703-cc55ef259c48" width=500>
</div>

</details>

## 定位
DataCopilot是PaddleMIX 2.0版本新推出的多模态数据处理工具箱，理念是把数据作为多模态算法的一部分参与迭代的全流程，让开发者根据特定任务以低代码量实现数据的基本操作。

## 核心概念 
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

## 使用案例
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

