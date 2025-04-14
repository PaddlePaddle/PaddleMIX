# Qwen2.5-Omni

## 1. 介绍
Qwen2.5-Omni 是阿里巴巴通义千问团队于2025年3月27日发布的新一代端到端多模态大模型。该模型能够无缝处理文本、图像、音频和视频等多种输入形式，并实时生成文本与自然语音合成输出，在多模态理解和生成任务上达到业界领先水平。其架构如下图所示，
<div align="center">
    <img src="https://github.com/user-attachments/assets/c2938338-2866-4b9a-b422-c43685f5603b" alt="Image 2" style="width: 50%;">
    <p style="color: #808080;"> Qwen-Omni 架构 </p>
</div>
Thinker负责处理和理解来自文本、音频、视频多模态的输入信息，Talker 以流式方式接收Thinker以自回归解码产生的高维表征与文本，输出离散化的语音token。Talker共享Thinker的全部历史上下文信息，从而使整个架构如同一个紧密协作的单一模型。

本仓库支持的权重：
| Model                       |
|-----------------------------|
| Qwen/Qwen2.5-Omni-7B        |

## 2. 环境安装
* PaddlePaddle Develop版本安装
```
pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/

```
* PaddleMIX安装
```
pip install -e .
pip install -e ppdiffusers
pip install -r requirements.txt
```
* Qwen2.5-Omni 环境依赖
```bash
# 使用conda环境安装ffmpeg
conda install -c conda-forge ffmpeg
pip install soundfile==0.13.1
```

若libcufft动态库缺失，尝试以下步骤，以确保动态链接库的路径正确。
```bash
find / -name "libcufft.so.11" 2>/dev/null
export LD_LIBRARY_PATH=/path/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH
```


## 3. 推理

### 3.1 图像理解与语音生成

```bash
python paddlemix/examples/qwen2_5_omni/image_inference.py \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --attn_implementation sdpa \
    --compute_dtype bfloat16 \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --question "What are in this image?" \
    --output_audio_file image_output.wav
```

### 3.2 视频理解与语音生成
```bash
python paddlemix/examples/qwen2_5_omni/video_inference.py \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --attn_implementation sdpa \
    --compute_dtype bfloat16 \
    --video_file paddlemix/demo_images/red-panda.mp4 \
    --question "What are in this video?" \
    --output_audio_file video_output.wav
```

## 4. 回答展示

### 4.1 图像理解

```
input:
['<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>What are in this image?<|im_end|>\n<|im_start|>assistant\n']

output:
["Well, in this image, there's a red panda. It's got that cute reddish-brown fur with white markings on its face. It's resting its head on a wooden box, and it looks like it's in a natural setting with some trees and greenery in the background. What do you think about red pandas? They're really interesting animals, aren't they?"]
```
### 4.2 视频理解
```
input:
['<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>What are in this video?<|im_end|>\n<|im_start|>assistant\n']

output:
["In the video, there are two red pandas. One is on a tree branch, and the other is on the ground. They seem to be playing with some kind of food or toy that's hanging from the tree. It's really cute to watch them interact. What do you think about red pandas?"]
```

## 参考文献
```
@article{Qwen2.5-Omni,
  title={Qwen2.5-Omni Technical Report},
  author={Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin},
  journal={arXiv preprint arXiv:2503.20215},
  year={2025}
}
```