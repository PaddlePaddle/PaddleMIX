# R1-MLLM
## 简介
本仓库基于Paddle实现了GRPO算法微调Qwen2.5-VL、Qwen2-VL视觉语言大模型，并支持指向性目标检测任务 (Referring Expression Comprehension)、计数问题 (Item Counting)、几何推理 (Geometry Reasoning)问题。


本仓库支持的权重
| Model                       |
|-----------------------------|
| Qwen/Qwen2-VL-2B-Instruct   |
| Qwen/Qwen2-VL-7B-Instruct   |
| Qwen/Qwen2.5-VL-3B-Instruct |
| Qwen/Qwen2.5-VL-7B-Instruct |

## 安装
1）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/develop?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）pip install math_verify

注意：Python需要使用3.10及以上版本。


## 数据准备

### 指向性目标检测任务

* 下载PaddleMIX团队整理好的数据集：
```bash
 https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/r1_mllm/REC.tar
```

或者分别下载原始数据集：

* 下载 [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip)  并且解压到指定路径PaddleMIX下的data/coco目录.
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/refcoco/train2014.tar
```

* 下载 [RefGTA](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/refgta.zip) 并解压到data/refgta目录。

* 下载 [RefCOCO/+/g and RefGTA Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) 解压放置PaddleMIX/data/rec_jsons_processed目录下 (RefGTA 域外测试数据,用于泛化性测试).


### 计数任务

* 下载PaddleMIX团队整理好的数据集：
```bash
 https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/r1_mllm/Counting.tar
```

或者分别下载原始数据集：

* 下载 [CLEVR-70K-Counting](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train) 训练数据集，修改your_path为你的实际安装路径路径。例如data/clevr_cogen_a_train
```bash
huggingface-cli download --resume-download leonardPKU/clevr_cogen_a_train --local-dir data/clevr_cogen_a_train --repo-type="dataset"
```

* 下载测试集 [SuperCLEVR-200](https://huggingface.co/datasets/tobiaslee/Super_clevr200/resolve/main/subsplit.tgz) 并解压到data/superclevr_200目录。

* 下载测试集标签 [superclevr_test200_counting_problems](https://github.com/Deep-Agent/R1-V/blob/main/src/eval/prompts/superclevr_test200_counting_problems.jsonl) 放置data目录下


### 几何推理任务

* 下载PaddleMIX团队整理好的数据集：
```bash
 https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/r1_mllm/GEO.tar
```

或者分别下载原始数据集：

* 下载 [GEOQA-8k](https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K) 到data/GEOQA_R1V_Train_8K 目录。
```bash
huggingface-cli download --resume-download leonardPKU/GEOQA_R1V_Train_8K --local-dir data/GEOQA_R1V_Train_8K --repo-type="dataset"
```
* 下载 [GEO170K](https://huggingface.co/datasets/Luckyjhg/Geo170K) 测试集 到data/GEOQA_R1V_Train_8K目录
```bash
huggingface-cli download --resume-download Luckyjhg/Geo170K --local-dir data/Geo170K --repo-type="dataset"

unzip data/Geo170K/images.zip -d data/Geo170K
```
 并解压到data/GEOQA_R1V_Train_8K目录
* 下载测试集标签 [geoqa_test_prompts](https://github.com/Deep-Agent/R1-V/blob/main/src/eval/prompts/geoqa_test_prompts.jsonl) 放置data目录下


## 指向性目标检测效果展示
### 性能指标
固定随机种子，从验证集中抽取500条数据测试，结果如下：

| Model                                | refcoco val|  refcoco+ val | refcocog val | RefGTA |
|--------------------------------------|------------|---------------|--------------|--------|
|  Qwen2.5-VL-3B-Instruct              |88.60%      |79.60%         |  81.80%      | 71.80% |
|  R1-Qwen2.5-VL-3B-Instruct(500steps) |88.40%      |83.60%         |  81.80%      | 74.60% |

### 训练曲线
![Image](https://github.com/user-attachments/assets/9df169fb-7fda-4156-8d62-d8baedf0f5f3)

### 训练回答样例
```
------------- Accuracy reward: 1.0 -------------
<think>
The bounding box describes the large, white vehicle on the street. The vehicle is large in size and can be identified by its white color. It is on the street, suggesting it is in motion or performing a task. Given the context of the other bounding boxes, this is likely a transport mode used over longer distances, such as a bus or truck.
</think>

<answer>
[352.14, 33.94, 639.59, 224.86]
</answer>

Solution: [352.14, 33.94, 639.59, 224.86]
------------- Format reward: 1 -------------
<think>
The bounding box describes the large, white vehicle on the street. The vehicle is large in size and can be identified by its white color. It is on the street, suggesting it is in motion or performing a task. Given the context of the other bounding boxes, this is likely a transport mode used over longer distances, such as a bus or truck.
</think>

<answer>
[352.14, 33.94, 639.59, 224.86]
</answer>
```

## 训练命令

```bash
# 八卡训练指向性目标检测 GRPO
bash paddlemix/examples/r1_mllm/scripts/run_grpo_rec.sh

# 八卡训练计数问题 GRPO
bash paddlemix/examples/r1_mllm/scripts/run_grpo_counting.sh

# 八卡训练几何推理问题 GRPO
bash paddlemix/examples/r1_mllm/scripts/run_grpo_geometry.sh
```

## 测试命令
```bash
# test baseline refcoco, 如果在V100机器上使用请加入传参 --dtype "float16"
python paddlemix/examples/r1_mllm/eval/test_rec.py \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --method "baseline" \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_root "data/rec_jsons_processed" \
    --image_root "data/coco" \
    --test_datasets refcoco_val refcocop_val refcocog_val \
    --batch_size 32 \
    --sample_num 500 \
    --steps 300 \
    --seed 42

# test r1 refgta
python paddlemix/examples/r1_mllm/eval/test_rec.py \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --method "r1" \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_root "data/rec_jsons_processed" \
    --image_root "data/refgta" \
    --test_datasets refgta_subsample \
    --batch_size 32 \
    --sample_num 500 \
    --steps 300 \
    --seed 42

# test r1 counting
python paddlemix/examples/r1_mllm/eval/test_r1-v.py \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --method "r1" \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_root "data/" \
    --image_root "data/superclevr_200/" \
    --test_datasets superclevr_test200_counting_problems \
    --batch_size 32 \
    --steps 500 \
    --seed 42

# test r1 geoqa
python paddlemix/examples/r1_mllm/eval/test_r1-v.py \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --method "r1" \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_root "data/" \
    --image_root "data" \
    --test_datasets geoqa_test_prompts \
    --batch_size 32 \
    --steps 500 \
    --seed 42
```

## 引用
```latex
@misc{shen2025vlmr1,
  author       = {Shen, Haozhan and Zhang, Zilun and Zhang, Qianqian and Xu, Ruochen and Zhao, Tiancheng},
  title        = {VLM-R1: A stable and generalizable R1-style Large Vision-Language Model},
  howpublished = {\url{https://github.com/om-ai-lab/VLM-R1}},
  note         = {Accessed: 2025-02-15},
  year         = {2025}
}

@misc{chen2025r1v,
  author       = {Chen, Liang and Li, Lei and Zhao, Haozhe and Song, Yifan and Vinci},
  title        = {R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than \$3},
  howpublished = {\url{https://github.com/Deep-Agent/R1-V}},
  note         = {Accessed: 2025-02-02},
  year         = {2025}
}
```
