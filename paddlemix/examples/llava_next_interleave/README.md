# LLaVA-NeXT-Interleave

## 1. 模型介绍

[LLaVA-NeXT-Interleave](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)是基于大规模语言模型 llava 的视觉语言模型。支持处理大型多模态模型中的多图像、视频和 3D 等场景。

LLaVA-NeXT-Interleave 可以在不同的多图像基准测试中取得与之前的 SoTA 相比领先的结果。（2）通过适当混合不同场景的数据，可以提高或保持之前单个任务的性能，保持了 LLaVA-NeXT 的单图像性能，并提高了视频任务的性能。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| lmms-lab/llava-next-interleave-qwen-0.5b  |
| lmms-lab/llava-next-interleave-qwen-7b  |
| lmms-lab/llava-next-interleave-qwen-7b-dpo  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("lmms-lab/llava-next-interleave-qwen-0.5b")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

版本要求：paddlenlp>=3.0.0b2

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本。


## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

### 多轮对话启动
```bash
# llava
python paddlemix/examples/llava_next_interleave/run_predict_multiround.py  \
--model-path "lmms-lab/llava-next-interleave-qwen-7b" \
--image-file "paddlemix/demo_images/twitter3.jpeg" "paddlemix/demo_images/twitter4.jpeg" \
```
可配置参数说明：
  * `model-path`: 指定llava系列的模型名字或权重路径
  * `image-flie` :输入图片路径或url，默认None。



输入图片：上述case

```
USER: Please write a twitter blog post with the images.
ASSISTANT: ✨ Launch! 🚀✨ The sky is alight with the brilliance of a rocket's ascent. The fiery trail of the rocket cuts through the darkness, a testament to human ingenuity and the relentless pursuit of exploration. The water below mirrors the spectacle, its surface rippling with the reflection of the celestial display. This moment captures the awe-inspiring power of technology and the boundless possibilities it holds for our future. #SpaceExploration #RocketLaunch
```


### 参考文献
```BibTeX
@misc{li2024llavanext-interleave,
	title={LLaVA-NeXT: Tackling Multi-image, Video, and 3D in Large Multimodal Models},
	url={https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/},
	author={Li, Feng and Zhang, Renrui and Zhang, Hao and Zhang, Yuanhan and Li, Bo and Li, Wei and Ma, Zejun and Li, Chunyuan},
	month={June},
	year={2024}
}
```
