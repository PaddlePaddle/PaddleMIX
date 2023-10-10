# VisualGLM

## 1. 模型简介

VisualGLM-6B 是一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。

VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。 关于VisualGLM的更多信息请参考[VisualGLM](https://github.com/THUDM/VisualGLM-6B/tree/main)。


## 2. 快速使用
`run_predict.py`脚本展示了使用VisualGLM的方法，可以执行以下命令进行启动。其中参数 `pretrained_name_or_path` 用于指定 MiniGPT4 的保存目录或模型名称。

```
python run_predict.py \
    --pretrained_name_or_path "THUDM/visualglm-6b" \
    --image_path "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"

```

下图这个示例展示了在使用visualglm-6b时的效果：

输入图片：<center><img src="https://github.com/PaddlePaddle/PaddleNLP/assets/35913314/b609132e-1c06-4e4c-b5b3-7d945cfe8839" /></center>

输入文本：“写诗描述一下这个场景”

输出:
```
泰坦尼克号，浪漫而美丽。
男女主角手牵手，共舞于船头。
夕阳余晖洒落，风景如画。
他们的身影如此优美，令人陶醉。 海水翻涌，波涛汹涌。
船上的人们，沉浸在这美妙的时刻中。
爱情的力量，让他们更加坚定。
他们在大海上翱翔，享受着彼此的温暖。 电影的结束，意味着爱情的开始。
他们将永远铭记这段美好的日子。
在回忆里，他们会珍惜这份爱。
```

输入文本：“这部电影的导演是谁？”

输出:
```
这部电影的导演是詹姆斯·卡梅隆(James Cameron)。
```

## 3. License 说明
VisualGLM-6B模型权重使用需要遵循清华大学发布的[Model License](https://github.com/THUDM/VisualGLM-6B/blob/main/MODEL_LICENSE.txt)。


## Reference
- [VisualGLM-6B Repo](https://github.com/THUDM/VisualGLM-6B/tree/main)
