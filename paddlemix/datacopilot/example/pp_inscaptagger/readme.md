
# PP-InsCapTagger

## 方案简介

PP-InsCapTagger(Instance Capability Tagger) 是 DataCopilot 基于 PaddleMIX 实现的数据集行为标签模型，用于为多模态数据实例能力打标，通过实例能力分布对数据集进行优化，可以提高模型训练效率，为数据集分析和评价提供了一种高效的方案。
结合模型推理打标结果对LLaVA SFT数据集进行优化，可以**提高LLaVA模型SFT阶段50%的训练效率**。

数据实例能力标签：在多模态任务中，每条数据都可以抽象出一种或多种能力，在训练时，模型会从这些数据中学习并增强自身对应的能力，如下图。为了评价和优化数据集，我们可以通过模型为每条多模态数据在模型训练中贡献的实例能力进行打标，并根据打标结果中数据实例能力分布进行数据集的优化，进而提升模型的训练效率。

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2a8931f-ce24-47c5-9970-b42031bb28c5" align="middle" width = "800" />
</p>

PP-InsCapTagger 基于 PaddleMix 进行训练，使用 `llava-v1.6-7b` 模型作为 `base` 模型。数据集使用多模态数据 [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) 的部分图片和多轮对话内容，并通过 GPT-4o 为每一条数据的实例能力进行打标，并将打标结果作为该条数据的 `tags` 属性进行保存，然后使用 DataCopilot 实现数据集的高效预处理，结合原始多轮对话内容和 `tags` 结果重构数据集的 `question` 和 `answer`。

PP-InsCapTagger 部分训练和推理的细节可以参考AI Studio 项目：[基于PaddleMIX的数据集行为标签分类器训推实例](https://aistudio.baidu.com/projectdetail/7917712)

## 模型使用示例

本项目提供 PP-InsCapTagger 使用脚本 `inference.py`, 通过`single_data`和`json_data`两种推理模式，可以分别实现以图像-文本对输入的单条样本推理 和 以`json`文件输入的批量数据推理。

### 单样本推理:

输入图片：<center><img src="https://github.com/user-attachments/assets/1c2fec64-3c94-4782-bc85-ccb083c1f4b2" width = "250"/></center>

输入多轮对话：

```
Q: What animal is in the image? A: The image features a dog.
Q: What color are the dog's eyes? A: The dog has blue eyes.
Q: Where is the dog situated in the image? A: The dog is situated inside a vehicle, on a front passenger seat.
```

```bash
# PaddleMIX根目录下执行
python paddlemix/datacopilot/example/pp_inscaptagger/inference.py \
single_data \
-m paddlemix/PP-InsCapTagger \
-image https://paddlenlp.bj.bcebos.com/models/community/paddlemix/PP-InsCapTagger/demo.jpg \
-qa "What animal is in the image?" "The image features a dog." \
    "What color are the dog's eyes?" "The dog has blue eyes." \
    "Where is the dog situated in the image?" "The dog is situated inside a vehicle, on a front passenger seat."
```

其中，`-m`表示模型所用权重路径，当值为`paddlemix/PP-InsCapTagger`时，会自动下载`PP-InsCapTagger`模型到本地；`-image`表示输入的图像地址(本地地址\http链接)；`-qa`表示输入的多轮对话内容，以空格分隔。

### 批量数据推理:

```bash
# PaddleMIX根目录下执行
python paddlemix/datacopilot/example/pp_inscaptagger/inference.py \
json_data \
-m paddlemix/PP-InsCapTagger \
-d path/to/your/data.json \
-k 0 \
-o path/to/your/output-dir
```
其中，`path/to/your/data.json` 为输入的批量数据文件路径，格式如下：

```json
[
    {
        "image": "http://ecx.images-amazon.com/images/I/51ntbts0gmL.jpg",
        "conversations": [
            [
                "<image>\nWhat is the genre of this book?",
                "Literature & Fiction"
            ],
            [
                "What is the color of this book?",
                "Red and black"
            ]

        ]
    },
    {
        "image": "http://ecx.images-amazon.com/images/I/51cc3XrLevL.jpg",
        "conversations": [
            [
                "<image>\nWhat is the title of this book?",
                "Beyond Bigger Leaner Stronger: The Advanced Guide to Building Muscle, Staying Lean, and Getting Strong (The Build Muscle, Get Lean, and Stay Healthy Series)"
            ]
        ]
    }
]
```
`-k`表示脚本批量处理的起始位置为第k个chunk的数据，默认为0，当处理中断时可以更改处理起始位置；`path/to/your/output-dir`表示处理结果json文件保存的位置，所有chunk的处理结果分别保存在对应的json文件中，命名为`tagger_{i:05}.json`。

## 标签使用案例

LLaVA v1.5模型SFT阶段训练时，使用的指令微调数据集为[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150)中llava_v1_5_mix665k数据集，该数据集为多个数据集混合而成，相比于预训练数据集，该数据集规模更大，同时在实例能力分布上也存在较大的差异。为了优化该数据集的实例能力分布，进而提高模型训练效率，我们使用PP-InsCapTagger对数据集进行打标，并统计标签分布。

使用PP-InsCapTagger对llava_v1_5_mix665k数据集进行打标，可以得到7913个标签，对数量最多的前100个标签分布进行可视化，可以看出标签分布存在较大的差异，如下图所示：

<details>
<summary>See</summary>
<center><img src="https://github.com/user-attachments/assets/48e30848-fe18-4e1a-a9a5-6c6f18ad9029" width = "300"/></center>
</details>


为了对llava_v1_5_mix665k数据集进行优化，我们使用PP-InsCapTagger打标的标签结果对数据集进行筛选，**首先确定出能够覆盖80%数据的单条数据的标签数量N，然后在数据集标签集合中选出标签数量占比前0.7%的标签作为一个筛选集合R，对于llava_v1_5_mix665k数据集中的每条数据，如果该条数据标签数量小于N，且该条数据的所有标签均在集合R中，则删除该条数据，否则保留该条数据**。通过该筛选策略，最终保留数据集规模为原始数据集的50%左右。

我们分别使用llava_v1_5_mix665k数据集和筛选后的数据集进行llava-1.5-7b SFT阶段训练，对比结果如下表所示：

| Version              | ScienceQA | TextVQA | VQAv2 | GQA   | mmmu  | mme            |
|:----------------------:|:-----------:|:---------:|:-------:|:-------:|:-------:|:----------------:|
| llava-1.5-7b <br> (paper) | 66.8 | 58.2 | 78.5 | 62.0 |  -  |  -  |
| llava-1.5-7b <br> (rerun) | 69.01 | 57.6 | 79.0 | 62.95 | 36.89 | 1521 <br> 323 |
| llava-1.5-7b <br> (tag 50%/our) | 70.24 | 57.12 | 78.32 | 62.14 | 37.11 | 1476 <br> 338 |

通过PP-InsCapTagger的打标和优化，50%数据集与原始数据集的训练效果基本持平，大大提高了模型训练效率。



## 引用
如果在你的工作中用到`PP-InsCapTagger`，请按照下面的方式引用：

<details>
<summary> bibtex </summary>

```bibtex

@software{PaddleMIX_Authors_Paddle_Multimodal_Integration,
author = {PaddleMIX Authors},
license = {Apache-2.0},
title = {{Paddle Multimodal Integration and eXploration}},
url = {https://github.com/PaddlePaddle/PaddleMIX}
}

@software{Lv_Instance_Capability_Tagger_2024,
author = {Lv, Wenyu and Huang, Kui and Zhao, Yian},
license = {Apache-2.0},
month = oct,
title = {{Instance Capability Tagger: Enhancing Multimodal Data Efficiency for Model Training}},
url = {https://github.com/lyuwenyu/PP-InsCapTagger},
version = {1.0},
year = {2024}
}
```
</details>
