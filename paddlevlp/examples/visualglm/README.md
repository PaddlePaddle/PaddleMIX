# VisualGLM

## 1. 模型简介

VisualGLM-6B 是一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。

VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。 关于VisualGLM的更多信息请参考[VisualGLM](https://github.com/THUDM/VisualGLM-6B/tree/main)。


## 2. 快速使用
`run_predict.py`脚本展示了使用VisualGLM的方法，可以执行以下命令进行启动。其中参数 `pretrained_name_or_path` 用于指定 MiniGPT4 的保存目录或模型名称。

```
python run_predict.py \
    -- pretrained_name_or_path "your minigpt4 path"

```

下图这个示例展示了在使用visualglm-6b时的效果：

输入图片：<center><img src="https://github.com/PaddlePaddle/Paddle/assets/35913314/d8070644-4713-465d-9c7e-9585024c1819" /></center>

输入文本：“写诗描述一下这个场景”

输出:
```
两个杯子，黑白相间，
一个放在桌子上，另一个放在咖啡杯上。
它们静静地坐着，
仿佛在讲述着什么故事。 一只猫和另一只猫，
彼此相依相伴，
似乎有着某种神秘的联系。
它们的黑白对比，
仿佛是一幅美丽的画，
让人不禁沉醉其中。 这两只杯子，
是一份温馨的礼物，
代表着爱和情感的温度。
它们在桌面上静静等待着，
期待着主人的到来，
让它们成为彼此的依靠。
```

输入文本：“这部电影的导演是谁？”

输出:
```
电影《猫与杯》由韩国著名导演李在均执导。
```

## 3. License 说明
VisualGLM-6B模型权重使用需要遵循清华大学发布的[Model License](./MODEL_LICENSE.txt)。


## Reference
- [VisualGLM-6B Repo](https://github.com/THUDM/VisualGLM-6B/tree/main)
