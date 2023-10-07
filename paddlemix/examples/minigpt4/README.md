# MiniGPT4

## 1. 模型简介

MiniGPT4 是一个具有图像理解能力的开源模型，其基于 Vicuna 大语言模型 以及 BLIP-2 中的VIT和Qformer模块进行训练，使得MiniGPT4 拥有类似于GPT4的非凡能力，例如详细的图像描述生成和从手写草稿创建网站。 此外 MiniGPT4 还具备一些的其他新的功能，包括根据给定图像写故事和诗歌，为图像中显示的问题提供解决方案，教用户如何根据食物照片做饭等。下图展示了MiniGPT4的模型结构， 更多信息请参考[MiniGPT4](https://arxiv.org/abs/2304.10592)。

<center><img src="https://github.com/PaddlePaddle/Paddle/assets/35913314/f0306cb6-4837-4f52-8f57-a0e7e35238f6" /></center>


## 2. 获取MiniGPT4 权重以及相关配置
这里可以分两步：1. 获取MiniGPT4权重；2. 获取相关配置，包括模型参数说明以及tokenizer相关文件等。
### 2.1 获取MiniGPT4权重
目前需要用户手动下载MiniGPT4权重和并转换为相应的 Paddle 版权重，为方便转换，本项目提供了相应的操作说明和转换脚本，详情请参考[MiniGPT4 权重下载和转换说明](./paddle_minigpt4_instrction.md)。

### 2.2 获取相关配置
下载相关的配置文件，这里提供了两版配置文件，请根据你的需要，点击下载即可。
|  files Aligned with MiniGPT4-7B  |  files Aligned with MiniGPT4-13B |
:-------------------------------------:|:-----------------------------------:
 [Download](https://paddlenlp.bj.bcebos.com/models/community/minigpt4-7b/minigpt4_7b.tar.gz)|[Download](https://paddlenlp.bj.bcebos.com/models/community/minigpt4-13b/minigpt4_13b.tar.gz) |


下载之后进行解压，请将其中相关文件放至 与 MiniGPT4 权重相同的目录中。


## 3. 模型预测
在下载和转换好上述模型权重之后，可执行以下命令进行模型预测。其中参数 `pretrained_name_or_path` 用于指定 MiniGPT4 的保存目录。

```
python run_predict.py \
    --pretrained_name_or_path "your minigpt4 path" \
    --image_path "you image path" \
    --decode_strategy "greedy_search" \
    --max_length 300 \
    --num_beams 1 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 1.0 \
    --length_penalty 0.0 \
    --temperature 1.0
```

你可以通过调整一些生成策略，来控制生成内容，例如一些常用的生成策略参数如下：
`pretrained_name_or_path`: 默认为MiniGPT4模型的存放目录.
`image_path`: 图片路径，支持传入URL或本地路径.
`decode_strategy`: 文本生成时的解码策略，目前支持'gready_search', 'sampling', 'beam_search'三种方式.
`max_length`: 生成序列的最大长度.
`num_beams`: 'beam_search'策略的束宽.
`top_p`: 'sampling'策略中top-p过滤的累积概率， 默认在区间[0,1)之间. 默认为1.0, 表示没有效果.
`top_k`: 'sampling'策略中，top-k过滤保留的最高概率token的数量， 默认为0， 表示没有效果.
`repetition_penalty`: 重复惩罚的参数，默认值为1，表示没有惩罚.
`temperature`: 'sampling'策略中，用于对下一个token概率进行建模的值。默认为1.0, 表示没有效果.


## 4. 效果展示

下图这个示例展示了在使用MiniGPT-7b时的效果：

输入图片：<center><img src="https://github.com/PaddlePaddle/Paddle/assets/35913314/d8070644-4713-465d-9c7e-9585024c1819" /></center>

输入文本：“describe this image”

输出:
```
The image shows two mugs with cats on them, one is black and white and the other is blue and white. The mugs are sitting on a table with a book in the background. The mugs have a whimsical, cartoon-like appearance. The cats on the mugs are looking at each other with a playful expression. The overall mood of the image is lighthearted and fun.###
```


## Reference
- [MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models](https://minigpt-4.github.io/)
