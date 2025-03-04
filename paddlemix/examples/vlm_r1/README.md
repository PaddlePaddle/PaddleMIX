# VLM-R1 
## 简介
自Deepseek-R1推出以来，许多研究工作都集中在对其的复现和改进上。如VLM-R1,R1-V。PaddleMIX团队决定启动复现R1在视觉语言大模型相关的研究工作，并在此基础上探索可能的优化与创新路径，推动视觉-语言大模型领域的进一步发展。


本仓库支持的权重
| Model                       |
|-----------------------------|
| Qwen/Qwen2.5-VL-3B-Instruct |


## 效果展示
## 性能指标
| Model | refcoco|  refcoco+  | refcocog | RefGTA | 
|------|--------|------------|-----------|--------|
|  Qwen2.5-VL-3B-Instruct   |xx% |xx%     |  xx%  | xx% |
|  R1-Qwen2.5-VL-3B-Instruct |xx |xx%      |  xx%  | xx% |

### 训练曲线
![Image](https://github.com/user-attachments/assets/82e253e2-69aa-4538-ad37-37caa8450b0c)
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

## 数据准备
> 1. 下载 [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip)  并且解压到指定路径如data目录.

> 2. 下载 [RefGTA] (https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/refgta.zip) 并解压

> 3. 下载 [RefCOCO/+/g and RefGTA Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) 解压放置PaddleMIX/data/目录下 (RefGTA 域外测试数据,用于泛化性测试).

> 4. 预处理标签文件:
```python
# 处理refcoco
python paddlemix/examples/vlm_r1/preprocess_refcoco.py \
    --json_path data/refcoco_train.json \
    --image_dir <your_image_root> \
    --output_path data/refcoco_train_new.json

# 处理refcoco+
python paddlemix/examples/vlm_r1/preprocess_refcoco.py \
    --json_path data/refcocop_train.json \
    --image_dir <your_image_root> \
    --output_path data/refcocop_train_new.json

# 处理refcocog
python paddlemix/examples/vlm_r1/preprocess_refcoco.py \
    --json_path data/refcocog_train.json \
    --image_dir <your_image_root> \
    --output_path data/refcocog_train_new.json

```

> 5. 修改配置文件中训练数据的路径 `paddlemix/examples/vlm_r1/src/open-r1-multimodal/data_config/rec.yaml` file.
```bash
datasets:
    - json_path: data/refcoco_train_new.json
    - json_path: data/refcocop_train_new.json
    - json_path: data/refcocog_train_new.json
```

## 训练命令
### GRPO

```bash
# 八卡训练
bash paddlemix/examples/vlm_r1/open-r1-multimodal/run_grpo_rec.sh
```

## 测试命令
```bash
python paddlemix/examples/vlm_r1/eval/test_rec.py \
    --method "r1" \
    --model_path "/path/to/model" \
    --data_root "/path/to/data" \
    --image_root "/path/to/coco" \
    --refcoco_val refcocop_val refcocog_val \
    --batch_size 32 \
    --sample_num 500 \
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
```