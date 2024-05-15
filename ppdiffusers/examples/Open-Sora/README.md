# hpcAI/Open-Sora前向推理支持
## 1. 简介

[hpcAI/Open-Sora](https://github.com/hpcAI/Open-Sora)为Sora复现版本之一, 其支持不同时长和分辨率的视频生成，并提供多任务推理，包括图生视频，视频拼接，视频编辑。

## 2. 环境准备

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装ppdiffusers以及必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.6.0 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.6.0版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入ppdiffusers目录
cd PaddleMIX/ppdiffusers

# 安装ppdiffusers，若提示权限不够，请在最后增加 --user 选项
pip install -e .

# 进入Open-Sora目录
cd examples/Open-Sora/

# 安装其他所需的依赖, 若提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

## 3. 前向推理
### 3.1 Text to video
运行以下命令，指定生成视频的帧数、分辨率以及提示词进行视频生成(推理相关参数设置详见`./utils/config_utils.py`)，以下例子提示词可从 `assets/texts`获取，可根据算力条件以生成更长分辨率更大的视频：
```bash
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python scripts/inference.py --prompt "A beautiful sunset over the city" --num-frames 16 --image-size 256 256
```
生成效果如下:
| **16×280×280**     | **16×224×400**        | **16×400×224**      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/926028e0-9f15-4fe0-ad9c-708f41ae9389) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/bcdd70ad-81b6-402b-a975-38e03069d5fd) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/c827e09c-aebc-4933-a350-558057690a04) |
| A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow [...]        | The vibrant beauty of a sunflower field. The sunflowers, with their bright yellow petals. [...]    | A majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow [...]       |

### 3.2 Image as condition

运行以下命令，以图像作为条件进行视频生成：
```bash
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python scripts/inference-long.py --num-frames 20 --image-size 256 256 --sample-name image-cond --prompt 'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/wave.png","mask_strategy": "0"}'
```
生成效果如下:
| **Prompts**     | **Image as condition**        | **20×256×256**      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A breathtaking sunrise scene. | ![demo](./assets/images/condition/wave.png) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/63aea8e6-cf89-431d-a14e-4895b491d198) |

### 3.3 Video connecting

运行以下命令，将首尾帧图像进行拼接，以获取对应视频：
```bash
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python scripts/inference-long.py --num-frames 16 --image-size 256 256 --sample-name connect --prompt 'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png;assets/images/condition/sunset2.png","mask_strategy": "0;0,1,0,-1,1"}'
```
生成效果如下:
| **Prompts**     | **First frame**        | **Last frame**        |  16×256×256      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A breathtaking sunrise scene. | ![demo](./assets/images/condition/sunset1.png) | ![demo](./assets/images/condition/sunset2.png) |![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/bc0d0dff-c045-459c-9f18-baf9751205da) |


### 3.4  Video extending and editting
此外支持以视频作为条件进行视频生成，包括视频扩展和视频编辑，运行脚本如下：
```bash
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
# video extending
python scripts/inference-long.py --num-frames 12 --image-size 240 240 --sample-name video_extend  --prompt 'A car driving on the ocean.{"reference_path": "./assets/videos/d0_proc.mp4","mask_strategy": "0,0,0,-6,6"}'

# video editting
python scripts/inference-long.py --num-frames 16 --image-size 256 256 --sample-name edit --prompt 'A cyberpunk-style car at New York city.{"reference_path": "./assets/videos/d0_proc.mp4","mask_strategy": "0,0,0,0,16,0.4"}'
```


**___Note: 多任务推理相关配置和原理详见[hpcAI/Open-Sora](https://github.com/hpcaitech/Open-Sora/blob/main/docs/config.md#advanced-inference-config)。___**

## 5. 参考资料
- [Open-Sora](https://github.com/hpcAI/Open-Sora)
