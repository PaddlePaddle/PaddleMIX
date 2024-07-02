### 视觉语言对话（Vision-Language-Chat）

#### 1. 应用介绍
输入图像或文字进行多轮对话，包括captions、grounding、视觉定位能力


#### 2. Demo

example:

```python

import paddle
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
paddle.seed(1234)
task = Appflow(app="image2text_generation",
                   models=["qwen-vl/qwen-vl-chat-7b"])
image= "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
prompt = "这是什么？"
result = task(image=image,prompt=prompt)

print(result["result"])

prompt2 = "框出图中公交车的位置"
result = task(prompt=prompt2)
print(result["result"])

```

输入图片：<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11" /></center>

prompt：“这是什么？”

输出:
```
这是一张红色城市公交车的图片，它正在道路上行驶，穿越城市。该区域似乎是一个住宅区，因为可以在背景中看到一些房屋。除了公交车之外，还有其他车辆，包括一辆汽车和一辆卡车，共同构成了交通场景。此外，图片中还显示了一一个人，他站在路边，可能是在等待公交车或进行其他活动。
```
prompt2：“框出图中公交车的位置”

输出:
```
<ref>公交车</ref><box>(178,280),(803,894)</box>
```
