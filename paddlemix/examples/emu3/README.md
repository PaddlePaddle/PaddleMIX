# Emu3

## 1. 模型介绍

[Emu3](https://github.com/baaivision/Emu3) 是北京智源人工智能研究院推出的一款原生多模态世界模型，采用智源自研的多模态自回归技术路径，在图像、视频、文字上联合训练，使模型具备原生多模态能力，实现图像、视频、文字的统一输入和输出。Emu3将各种内容转换为离散符号，基于单一的Transformer模型来预测下一个符号，简化了模型架构。其架构如下所示：

![overview](https://github.com/user-attachments/assets/77275c9b-21ea-4603-8d15-8e507fce5038)
> 注：图片引用自[Emu3](https://github.com/baaivision/Emu3/blob/main/assets/arch.png).


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| BAAI/Emu3-VisionTokenizer  |
| BAAI/Emu3-Gen  |
| BAAI/Emu3-Chat  |

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

### a. 视觉离散编码器
```bash
python paddlemix/examples/emu3/autoencode.py \
    --model_path="BAAI/Emu3-VisionTokenizer" \
    --image_path="paddlemix/demo_images/emu3_demo.png"
```

### b. 图像生成
```bash
# 使用图像生成需要超过40G显存，否则会报错。
python paddlemix/examples/emu3/run_generation_inference.py \
    --model_path="BAAI/Emu3-Gen" \
    --vq_model_path="BAAI/Emu3-VisionTokenizer" \
    --prompt="a portrait of young girl." \
    --ratio="1:1" \
    --height=720 \
    --width=720 \
    --dtype="bfloat16"
```

### c. 多模态理解
```bash
python paddlemix/examples/emu3/run_understanding_inference.py \
    --model_path="BAAI/Emu3-Chat" \
    --vq_model_path="BAAI/Emu3-VisionTokenizer" \
    --image_path="paddlemix/demo_images/emu3_demo.png" \
    --question="Please describe the image briefly" \
    --max_new_tokens=512 \
    --dtype="bfloat16"
```

## 4 结果展示

### 4.1 离散视觉编码器
![vae](https://github.com/user-attachments/assets/95ac436e-f5b3-4d67-9468-f09480102482)
### 4.2 图像生成
![image_generation](https://github.com/user-attachments/assets/73599876-e284-4d16-ac3c-b93081ea550c)
### 4.3 多模态理解
![emu3_demo](../../demo_images/emu3_demo.png)

```
User: Please describe the image briefly

Assistant: The image features a photograph of a dog with a background of green grass and yellow flowers. The dog appears to be a collie, characterized by its long, fluffy fur that is predominantly brown and white. The dog's fur is particularly long around its neck and chest, giving it a distinctive and endearing appearance. The dog's ears are perked up, and it has a bright, happy expression on its face, with its mouth open and tongue slightly visible, suggesting that it might be painting or smiling. The dog's eyes are bright and alert, and it seems to be looking directly at the camera, creating a sense of connection with the viewer.

Above the dog's image, there is a quote in white text that reads: "My dogs have been the reason I have woken up every single day of my life with a smile on my face. Jennifer Skiff." The quote is attributed to Jennifer Skiff, and it suggests that the dog has had a significant positive impact on the person's life, making them smile every day.

In the bottom right corner of the image, there is a small logo for "GoodDogInABox.com," indicating that this image might be part of a series or campaign related to dogs and their impact on people's lives.

The overall composition of the image is simple yet effective, focusing on the dog's joyful expression and the accompanying text that conveys a personal and heartfelt message about the relationship between dogs and their owners. The use of green grass and yellow flowers in the background adds a natural and serene touch to the image, enhancing the overall positive and uplifting mood.
```

## 参考文献
```BibTeX
@article{wang2024emu3,
  title={Emu3: Next-Token Prediction is All You Need},
  author={Wang, Xinlong and Zhang, Xiaosong and Luo, Zhengxiong and Sun, Quan and Cui, Yufeng and Wang, Jinsheng and Zhang, Fan and Wang, Yueze and Li, Zhen and Yu, Qiying and others},
  journal={arXiv preprint arXiv:2409.18869},
  year={2024}
}
```
