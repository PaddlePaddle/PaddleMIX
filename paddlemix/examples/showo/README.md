# Show-o

## 1. 模型介绍

[Show-o](https://github.com/showlab/Show-o) 提出了一种统一的Transformer模型，该模型统一了多模态理解和生成。与完全自回归模型不同，Show-o结合了自回归和（离散）扩散建模，以自适应地处理各种混合模态的输入和输出。这一统一模型灵活支持广泛的视觉-语言任务，包括视觉问答、文本到图像的生成、文本指导的图像修复/外推以及混合模态生成。在各种基准测试中，Show-o展现出了与现有专为理解或生成设计的、参数相当或更多的单个模型相比，可比或更优的性能。这充分凸显了其作为下一代基础模型的潜力。
![showo_overview](https://github.com/user-attachments/assets/56745449-5ea2-4794-9f18-41e9a8a0f223)
> 注：上图为Show-o模型的整体架构图，引用自论文。

## 2 环境准备
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本3.0.0b2及以上**
- **paddlenlp == 3.0.0b3**

```
# 安装示例
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```
> 注：
> * 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
> * (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 推理预测

### a. 多模态理解预测
```bash
python3 paddlemix/examples/showo/inference_mmu.py \
  config=paddlemix/examples/showo/configs/showo_demo_w_clip_vit_512x512.yaml \
  max_new_tokens=100 dtype=float16 \
  mmu_image_root=paddlemix/examples/showo/mmu_validation \
  question='Please describe this image in detail. *** Do you think the image is unusual or not?'
```

### b. 多模态生成预测
```bash
python paddlemix/examples/showo/inference_t2i.py \
  config=paddlemix/examples/showo/configs/showo_demo_512x512.yaml \
  batch_size=1 validation_prompts_file=paddlemix/examples/showo/validation_prompts/showoprompts.txt \
  guidance_scale=1.75 generation_timesteps=16 \
  mode='t2i' dtype="float16"\

```
## 4 推理结果展示
### a. 多模态理解预测
```
User: Please describe this image in detail.
```
![understanding](./mmu_validation/dog.png)
```
Answer: The image is unusual because it features a living room scene with a couch and a chair,
but instead of a typical living room setting,
it is set in an underwater environment.
This is not a common sight, as living rooms are usually indoors and not designed to be submerged in water.
The presence of a couch and a chair in an underwater setting is unexpected and adds an element of surrealism to the scene.
```
### b. 多模态生成预测
![t2i](https://github.com/user-attachments/assets/9ed43c88-f8ce-4a0e-9503-b224a550b79e)

## 参考文献
```BibTeX
@article{xie2024showo,
  title={Show-o: One Single Transformer to Unify Multimodal Understanding and Generation},
  author={Xie, Jinheng and Mao, Weijia and Bai, Zechen and Zhang, David Junhao and Wang, Weihao and Lin, Kevin Qinghong and Gu, Yuchao and Chen, Zhijie and Yang, Zhenheng and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2408.12528},
  year={2024}
}
```
