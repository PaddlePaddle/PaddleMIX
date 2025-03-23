# 思维链在文生图的应用(Image Generation by Chain-of-Thought)
## 简介

思维链（CoT）是一种提示工程技术，旨在通过以模仿人类推理的方式构建输入提示，引导模型在复杂任务上的能力。本仓库通过加入奖励模型对文生图生成过程进行监督、引导，达到Inference-time Scaling的效果，实现了CoT在文生图任务上的应用。


本仓库支持Cot的模型权重
| Model                                           |
|-------------------------------------------------|
| stabilityai/stable-diffusion-xl-base-1.0        |
| stabilityai/stable-diffusion-2-1                |
| stabilityai/stable-diffusion-3-medium-diffusers |
| stabilityai/stable-diffusion-3.5-medium         |

## 安装
参考[PaddleMIX 安装教程](../../../README.md#3-‼️安装paddlepaddle)

## 运行
在 PaddleMIX/ppdiffusers/examples/cot/ 下，下载奖励模型权重：
```bash
wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/ImageReward.pdparams
```

```bash
#!/bin/bash
python run_sd_cot_predict.py \
--use_smc \
--model_name='stabilityai/stable-diffusion-xl-base-1.0' \
--lmbda=10.0 \
--resample_frequency=20 \
--resample_t_start=20 \
--resample_t_end=80 \
--num_particles=4 \
--potential_type=max \
--prompt "a photo of a train below an airplane" \
--metrics_to_compute=ImageReward \
--num_inference_steps=100 \

```

**参数说明**
```
--use_smc: 是否使用粒子蒙特卡洛方法,实现cot
--model_name: 模型名称
--lmbda: 奖励模型权重
--resample_frequency: 重采样频率
--resample_t_start: 重采样开始step
--resample_t_end: 重采样结束step
--num_particles: selection number,当使用smc时，num_particles需要大于1。
--potential_type: 奖励策略，支持max,diff,add
--prompt: 输入提示
--metrics_to_compute: 奖励模型
--num_inference_steps: SD模型推理步数

```


## 示例

### SDXL

```bash
#!/bin/bash
python run_sd_cot_predict.py \
--use_smc \
--model_name='stabilityai/stable-diffusion-xl-base-1.0' \
--lmbda=10.0 \
--resample_frequency=20 \
--resample_t_start=20 \
--resample_t_end=80 \
--num_particles=4 \
--potential_type=max \
--prompt "a photo of a train below an airplane" \
--metrics_to_compute=ImageReward \
--num_inference_steps=100
```

<div align="center">

| Text prompt | not cot | num_particles=2 | num_particles=4 |
|:----:|:----:|:----:|:----:|
| a photo of a train below an airplane|![00000](https://github.com/user-attachments/assets/096058b1-88b5-45a1-9f5b-2fed3d44691f) |![00000 (1)](https://github.com/user-attachments/assets/0e26af2c-99bb-4ab6-a8db-eb60dd16c027) |![00000 (2)](https://github.com/user-attachments/assets/7a0b750e-8f04-4015-b107-a57923269223) |
</div>



### SD2-1

```bash
#!/bin/bash
python run_sd_cot_predict.py \
--use_smc \
--model_name='stabilityai/stable-diffusion-2-1' \
--lmbda=10.0 \
--resample_frequency=20 \
--resample_t_start=20 \
--resample_t_end=80 \
--num_particles=4 \
--potential_type=max \
--prompt "a photo of a white pizza and a green umbrella" \
--metrics_to_compute=ImageReward \
--num_inference_steps=50
```

<div align="center">

| Text prompt | not cot | num_particles=2 | num_particles=4 |
|:----:|:----:|:----:|:----:|
|a photo of a white pizza and a green umbrella |![00000](https://github.com/user-attachments/assets/6579748e-edea-4413-81c5-f6299a515cc6)  | ![00000 (1)](https://github.com/user-attachments/assets/5e2713fa-5b6a-49af-be48-d7221ccc139b) |![00000 (2)](https://github.com/user-attachments/assets/b1c12587-e2de-47b3-bbd5-1f5fbffed78f) |

</div>



### SD3
```bash
#!/bin/bash
python run_cot_predict.py \
--use_smc \
--model_name='stabilityai/stable-diffusion-3-medium-diffusers' \
--lmbda=10.0 \
--resample_frequency=20 \
--resample_t_start=20 \
--resample_t_end=28 \
--num_particles=4 \
--potential_type=max \
--prompt "a photo of four knifes" \
--metrics_to_compute=ImageReward \
--num_inference_steps=28
```

<div align="center">

| Text prompt | not cot | num_particles=2 | num_particles=4 |
|:----:|:----:|:----:|:----:|
| a photo of four knifes|![00000](https://github.com/user-attachments/assets/d43545bb-4eac-42c5-88b5-252d27d89774) |![00000 (1)](https://github.com/user-attachments/assets/207101d4-20fb-4386-835f-e2a18c65544d) | ![00000 (2)](https://github.com/user-attachments/assets/e3ff5a38-3589-4947-a813-35af6090be5e)|
</div>



# Citation

```bibtex
@misc{singhal2025generalframeworkinferencetimescaling,
      title={A General Framework for Inference-time Scaling and Steering of Diffusion Models},
      author={Raghav Singhal and Zachary Horvitz and Ryan Teehan and Mengye Ren and Zhou Yu and Kathleen McKeown and Rajesh Ranganath},
      year={2025},
      eprint={2501.06848},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.06848},
}
