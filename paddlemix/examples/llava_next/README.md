# LLaVA

## 1. 模型介绍

[LLaVA-NeXT-Interleave](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)是基于大规模语言模型 llava 的视觉语言模型。支持处理大型多模态模型中的多图像、视频和 3D 等场景。

LLaVA-NeXT-Interleave 可以在不同的多图像基准测试中取得与之前的 SoTA 相比领先的结果。（2）通过适当混合不同场景的数据，可以提高或保持之前单个任务的性能，保持了 LLaVA-NeXT 的单图像性能，并提高了视频任务的性能。


本仓库提供paddle版本的llava-next-interleave-qwen-7b、llava-next-interleave-qwen-0.5b、llava-next-interleave-qwen-7b-dpo三个模型权重。


## 2 环境准备
- **python >= 3.8**
- **paddlenlp >= 3.0**

## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

### 多轮对话启动
```bash
# llava
python paddlemix/examples/llava_next/run_predict.py  \
--model-path "paddlemix/llava_next/llava-next-interleave-qwen-7b" \
--image-file "https://bj.bcebos.com/v1/paddlenlp/models/community/Llava-Next/twitter3.jpeg https://bj.bcebos.com/v1/paddlenlp/models/community/Llava-Next/twitter4.jpeg" \
```
可配置参数说明：
  * `model-path`: 指定llava系列的模型名字或权重路径 ，支持 'paddlemix/llava_next/llava-next-interleave-qwen-7b','paddlemix/llava_next/llava-next-interleave-qwen-7b-dpo','paddlemix/llava_next/llava-next-interleave-qwen-0.5b'
  * `image-flie` :输入图片路径或url，默认None。



输入图片：上述case

```
USER: Please write a twitter blog post with the images.
ASSISTANT: Just witnessed a stunning rocket launch! The bright light illuminated the night sky, leaving a trail of smoke in its wake. The rocket's fiery glow contrasted beautifully with the dark backdrop of the night. The sound of the engines echoed through the air, adding to the excitement of the moment. This is truly a sight to behold! #rocketlaunch #spaceexploration #astronauts
```
