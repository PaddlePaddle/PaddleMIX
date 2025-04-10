# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 是 Qwen 团队推出的一个专注于视觉与语言（Vision-Language, VL）任务的多模态大模型。它旨在通过结合图像和文本信息，提供强大的跨模态理解能力，可以处理涉及图像描述、视觉问答（VQA）、图文检索等多种任务。Qwen2-VL通过引入创新性的技术如 Naive Dynamic Resolution 和 M-RoPE，以及深入探讨大型多模态模型的潜力，显著地提高了多模态内容的视觉理解能力。

PaddleMIX团队基于`Qwen2-VL-2B-Instruct`设计了专门针对文档理解类任务的特色模型[PP-DocBee](../ppdocbee/)，欢迎使用。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| Qwen/Qwen2-VL-2B-Instruct  |
| Qwen/Qwen2-VL-7B-Instruct  |
| Qwen/Qwen2-VL-72B-Instruct  |
| Qwen/Qwen2-VL-2B  |
| Qwen/Qwen2-VL-7B  |
| Qwen/Qwen2-VL-72B  |
| Qwen/QVQ-72B-Preview  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备
1）[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求是3.0.0b2或develop版本**
```bash
# 提供三种 PaddlePaddle 安装命令示例，也可参考PaddleMIX主页的安装教程进行安装

# 3.0.0b2版本安装示例 (CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Develop 版本安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# sh 脚本快速安装
sh build_paddle_env.sh
```

2）[安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **paddlenlp >= 3.0.0b3**

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

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。

## 3 推理预测

### a. 单图预测 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_vl/single_image_infer.py
```

### b. 多图预测 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_vl/multi_image_infer.py
```

### c. 视频预测 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_vl/video_infer.py
```

### d. batch推理 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_vl/single_image_batch_infer.py
```
### 模型推理支持分布式推理

```bash
# 2B
sh paddlemix/examples/qwen2_vl/shell/distributed_qwen2_vl_infer_2B.sh
# 7B
sh paddlemix/examples/qwen2_vl/shell/distributed_qwen2_vl_infer_7B.sh
# 72B
sh paddlemix/examples/qwen2_vl/shell/distributed_qwen2_vl_infer_72B.sh
# 72B QVQ
sh paddlemix/examples/qwen2_vl/shell/distributed_qwen2_vl_infer_72B_QVQ.sh
```
> ⚠️注意："mp_degree"需要根据显卡数量"gpus"进行调整，例如2卡推理，则设置为2。

## 4 模型微调

### 4.1 小型示例数据集

PaddleMIX团队整理了`chartqa`和`LaTeX_OCR`数据集作为小型的示例数据集，下载链接为：

```bash
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/playground.tar # 1.0G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/LaTeX_OCR.tar # 1.7G
```
playground/目录下包括了图片目录`data/chartqa/`和标注目录`opensource_json/`，详见`paddlemix/examples/qwen2_vl/configs/demo_chartqa_500.json`。
LaTeX_OCR/目录下包括了图片目录和标注文件，详见`paddlemix/examples/qwen2_vl/configs/LaTeX_OCR.json`。
训练时只需修改对应shell脚本中的`meta_path`参数即可。如`meta_path="paddlemix/examples/qwen2_vl/configs/demo_chartqa_500.json"`。


### 4.2 大型公开数据集

大型的数据集选择6个公开的数据集组合，包括`dvqa`、`chartqa`、`ai2d`、`docvqa`、`geoqa+`、`synthdog_en`，详见`paddlemix/examples/qwen2_vl/configs/baseline_6data_330k.json`

PaddleMIX团队整理后的下载链接为：
```bash
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource_json.tar
```

注意：若先下载了示例数据集的`playground.tar`解压了，此处需删除后，再下载公开数据集的`playground.tar`并解压，opensource_json.tar需下载解压在playground/目录下，opensource_json 里是数据标注的json格式文件。

### 4.3 微调命令

注意：
1）此微调训练为语言模型微调，冻结视觉编码器而放开LLM训练。
2）默认总bs=32，每卡bs=2，gradient_accumulation_steps=2，默认分布式训练配置为"paddle sharding stage2"策略，对应于“torch DeepSpeed ZeRO-2"策略。在LaTeX_OCR数据集训练下，2B模型微调训练的显存大小约为18G，7B模型全量微调训练的显存大小约为50G。***若训练数据集平均分辨率较大时，显存会进一步增加。***
3）若默认训练配置下显存不足，可以调节训练shell脚本中的参数，如```PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}```改小每卡bs为1，以及选择"paddle sharding stage3"策略```--sharding="stage3"```。

```bash
# 2B (多张40G A卡 显存可运行2B模型)
sh paddlemix/examples/qwen2_vl/shell/baseline_2b_bs32_1e8.sh

# 2B lora (多张40G A卡 显存可运行2B模型)
sh paddlemix/examples/qwen2_vl/shell/baseline_2b_lora_bs32_1e8.sh

# 7B (多张80G A卡 显存可运行7B模型)
sh paddlemix/examples/qwen2_vl/shell/baseline_7b_bs32_1e8.sh

# 7B lora (多张80G A卡 显存可运行7B模型)
sh paddlemix/examples/qwen2_vl/shell/baseline_7b_lora_bs32_1e8.sh
```

注意：微调2b模型的运行示例如下：
![运行示例](../../demo_images/qwen2-vl-2b-lora-ft.png)

### 4.4 微调后使用
同按步骤3中的模型推理预测，只需将`paddlemix/examples/qwen2_vl/single_image_infer.py`中的`--model_path`参数修改为微调后的模型路径即可。

```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_vl/single_image_infer.py
```

### 4.5 自动并行的模型微调
#### 4.5.1 模型组网介绍

自动并行组网 [modeling_qwen2_vl_network.py](../../models/qwen2_vl/modeling_qwen2_vl_network.py) ，当前主要支持SFT和LoRA两种微调方式。
#### 4.5.2自动并行微调命令
```bash
# 2B (多张40G A卡 显存可运行2B模型)
sh paddlemix/examples/qwen2_vl/shell/auto_2b_bs32_1e8.sh

# 2B (多张40G A卡 显存可运行2B模型)
sh paddlemix/examples/qwen2_vl/shell/auto_2b_lora_bs32_1e8.sh
```
在sh脚本文件中，与手动并行主要的args配置区别：
```bash
  --enable_auto_parallel 1\
  --auto_parallel_resume_form_hybrid_parallel true \
  --use_intermediate_api true \
```
运行时，运行示例与4.3节一致。
#### 4.5.3 自动并行模型推理预测
需要将自动并行训练保存下来的多卡分布式权重进行合并，合并方法为：
```python
import paddle
import paddle.distributed as dist
ckpt_path='/path/for/dist_ckpt'# 你的自动并行权重路径文件夹(ex:work_dirs/auto_330k_2b_bs32_1e8/checkpoint-1000/dist_ckpt)
# offload=1, 参数 offload 到 CPU，减少显存占用
# prefix="model" 参数可用于过滤掉非模型参数，例如 optimizer 状态等
merged_state_dict = dist.checkpoint.load_state_dict.load_merged_state_dict(ckpt_path, offload=0, prefix="model")
paddle.save(merged_state_dict, 'model_state.pdparams')# 合并后的权重，与4.3手动并行一致
```

拿到合并后的权重之后，将合并后的权重放在/path/for/dist_ckpt文件夹下，在推理预测时，同步骤3当中的模型推理预测相同，将步骤3中对应的的infer文件的--model_path参数修改为合并后的权重路径所在的文件夹(/path/for/dist_ckpt)即可。

LoRA微调的权重合并方法类似，但合并之后需要通过python脚本将LoRA权重进行merge，merge完成之后才能用于推理，执行方式如下:
```sh
python paddlemix/examples/qwen2_vl/merge_lora_params.py --model_name_or_path Qwen/Qwen2-VL-2B-Instruct --lora_path  ./checkpoints/your_path --merge_model_path ./checkpoints/merged_model

```
### 5 高性能推理优化

[Paddle高性能推理优化后](../../../deploy/qwen2_vl/)，测试结果如下：

- 在 NVIDIA A800-80GB 上测试的单图端到端速度性能如下：

| model                  | Paddle Inference|    PyTorch   | Paddle 动态图 |
| ---------------------- | --------------- | ------------ | ------------ |
| Qwen2-VL-2B-Instruct   |      1.053 s     |     2.086 s   |   5.766 s   |
| Qwen2-VL-7B-Instruct   |      2.293 s     |     3.132 s   |   6.221 s   |



## 参考文献
```BibTeX
@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```
