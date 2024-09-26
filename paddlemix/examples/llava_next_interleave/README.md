# LLaVA-NeXT-Interleave

## 1. 模型介绍

[LLaVA-NeXT-Interleave](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)是基于大规模语言模型 llava 的视觉语言模型。支持处理大型多模态模型中的多图像、视频和 3D 等场景。

LLaVA-NeXT-Interleave 可以在不同的多图像基准测试中取得与之前的 SoTA 相比领先的结果。（2）通过适当混合不同场景的数据，可以提高或保持之前单个任务的性能，保持了 LLaVA-NeXT 的单图像性能，并提高了视频任务的性能。


本仓库提供paddle版本的llava-next-interleave-qwen-7b、llava-next-interleave-qwen-0.5b、llava-next-interleave-qwen-7b-dpo三个模型权重。


## 2 环境准备
- **python >= 3.8**
- <span style="color:red;">**paddlenlp >= 3.0**</span>
```
cd PaddleMIX/paddlemix/examples/llava_next_interleave
pip install -r requirement.txt

or 

pip install paddlenlp==3.0.0b0
```



## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

### 多轮对话启动
```bash
# llava
python paddlemix/examples/llava_next_interleave/run_siglip_encoder_predict.py  \
--model-path "paddlemix/llava_next/llava-next-interleave-qwen-7b" \
--image-file "paddlemix/examples/llava_next_interleave/demo_images/twitter3.jpeg" "paddlemix/examples/llava_next_interleave/demo_images/twitter4.jpeg" \
```
可配置参数说明：
  * `model-path`: 指定llava系列的模型名字或权重路径 ，支持 
  - paddlemix/llava_next/llava-next-interleave-qwen-7b
  - paddlemix/llava_next/llava-next-interleave-qwen-7b-dpo
  - paddlemix/llava_next/llava-next-interleave-qwen-0.5b
  - paddlemix/llava_next/llava-next-interleave-qwen-clip-encoder-7b
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
