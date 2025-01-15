# PaddleMIX MULLM WebUI

## 1. 简介
PaddleMIX MULLM_WebUI 是一个基于PaddleMIX套件的交互式平台，主要支持多模态理解任务的模型微调与推理功能。MULLM_WebUI 提供了丰富的可视化操作界面，支持用户进行模型微调、推理等操作。
![overview](./fig/overview.jpg)

#### 支持模型
| Model |Model Size |Inference | SFT | LoRA |
|-------|------------|-------|---|-----|
| qwen2_vl|2B/7B| ✅     | ✅   | ✅   |
| PPDocBee-2B-1129|2B | ✅     | ✅   | ✅ |

>* ✅: Supported
>* 🚧: In Progress
>* ❌: Not Supported

## 2. 安装
* 安装Paddle和PaddleMIX依赖

* 安装PaddleMIX MULLM WebUI依赖
```
pip install -r paddlemix/MULLM_WebUI/requirements.txt
```

## 3. 快速使用

### 3.1 启动
```
CUDA_VISIBLE_DEVICES=0 \
GRADIO_SHARE=1 \
GRADIO_SERVER_NAME=0.0.0.0 \
GRADIO_ANALYTICS_ENABLED=0 \
GRADIO_SERVER_PORT=8260 python paddlemix/MULLM_WebUI/run_web.py
```
### 3.2 使用教程
#### 3.2.1 新增数据集
##### 1) PaddleMIX官方中文数据集（部分）
* 为了方便大家进行训练，我们给出了使用 DataCopilot处理图片得到的高质量[文档QA数据集](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/datasets/pp-docbee/test_data.tar)，该数据集包含1700张图片，包含多个关于文章内容、图表等类型的问答对话。
* 在PaddleMIX下创建目录data, 将解压后到`test_data`到`./data`目录下, 并新建`dataset_info.json`并填入以下内容

```
{
    "test_data":{
        "file_name": "test_data/example.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}
```
##### 2) 新增Pokemon数据集
* 下载 [Pokemon](https://huggingface.co/datasets/llamafactory/pokemon-gpt4o-captions/tree/main) 数据集。Pokemon-gpt4o-captions 是一个基于精灵宝可梦的中英双语视觉问答数据集，其问答结果由gpt4o生成。其中中文问答数据共计833条，数据集大小80.8M。
* 放置中文数据集文件到 `./data/pokemon_gpt4o_zh/pokemon_gpt4o_zh.parquet`

* 运行转换数据集脚本
```
python paddlemix/MULLM_WebUI/scripts/convert_dataset.py \
    --data_dir ./data \
    --dataset_dir pokemon_gpt4o_zh \
    --file_name ./data/pokemon_gpt4o_zh/pokemon_gpt4o_zh.parquet
```
> 注：目前MULLM WebUI只支持单卡微调，为了达到更佳的训练效果，建议自己构建数据集或者按照[qwen2_vl ](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/qwen2_vl)样例中提供的脚本进行微调。
#### 3.2.2 模型微调
1) 模型选择

![模型选择](./fig/train_1.jpg)


2) 超参数设置
![超参数设置](./fig/train_2.jpg)


3) LoRA参数设置与模型训练
![模型训练](./fig/train_3.jpg)

#### 3.2.3 模型推理

1) 模型加载
![模型加载](./fig/chat_1.jpg)


2) 多模态理解
![多模态理解](./fig/chat_2.jpg)

## 4. 使用展示


1） 模型微调
![模型微调样例](./fig/example_train.jpg)


2） 模型推理
![模型推理样例](./fig/example_chat.jpg)

## 参考文献

```BibTeX
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```
