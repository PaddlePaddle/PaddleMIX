# 基于Hotshot-XL与DeepSeek模型的文生图优化

摘要：该项目旨在通过集成前沿的Hotshot-XL与DeepSeek模型，在将用户输入的Prompt转化为GIF动画的基础上，对动图进行自适应优化。具体地，基于推理大模型DeepSeek-R1的长思考能力，通过角色扮演，对于用户的prompt进行自动化拓展与美化，再通过复现的Hotshot-XL模型进行文生图操作，最后通过基于Gradio构建的网页进行GIF动画展示，可以清晰地发现动画质量得到了明显提升。


![gradio](https://github.com/user-attachments/assets/30dbc1bd-a344-4d94-b014-59b183feedd6)
图1: GIF动画优化前后对比，第一行的动画为原始动画，第二行的动画为优化后的动画

## 代码实现

### 安装PaddlePaddle
```
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

### 安装PaddleMIX 2.0
```
git clone https://github.com/PaddlePaddle/PaddleMIX
```
### 安装依赖
```
cd PaddleMIX/ppdiffusers
pip install -e . --user
cd examples/community/Hotshot-XL
pip install -r requirements.txt
```
### 转换模型
```
sh convert_pre.sh
pip install imageio
```
### 模型推理
生成原始GIF
```
python inference.py \
  --prompt="a smoking horse on the grassland, hd, high quality" \
  --seed 452 --precision f32 \
  --output="horse.gif"
```
![horse](https://github.com/user-attachments/assets/45571b9c-cc6f-4778-a85d-bc066b007a16)



将原始文本基于DeepSeek-R1进行拓展完善 （xxxx需要更换为自己的key）：
```
from openai import OpenAI
client = OpenAI(api_key="xxxx", base_url="https://api.deepseek.com/beta")

import argparse

parser = argparse.ArgumentParser(description='参数')
parser.add_argument('prompt', type=str, help='图片描述')
args = parser.parse_args()
prompt = args.prompt

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system",
         "content": "You are a picture beautifier. The user will provide you with a description of a picture, and you need to beautify this description from multiple aspects and output the refined picture description, which will be used for text - to - image generation. You only need to output the final prompt!"},
        {"role": "user", "content": prompt},
    ],
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
print(reasoning_content)
print("=================================================")
print(content)
```
生成拓展Prompt
```
python deepseek.py "Fish are flying in the sky."

Okay, the user wants me to beautify the description "Fish are flying in the sky." for a text-to-image prompt. Let me think about how to enhance this. First, I need to add more vivid details. Maybe specify the type of fish, like tropical or koi. Then, the sky – perhaps a clear azure sky with fluffy clouds. Lighting is important; maybe golden sunlight to make the colors pop. The fish could have iridescent scales shimmering in the light. Adding movement elements, like schools of fish gliding gracefully. The atmosphere should be surreal and magical. Maybe mention the environment below, like a serene meadow or mountains to contrast with the flying fish. Also, using words like "ethereal" and "dreamlike" to set the mood. Let me put it all together cohesively.
=================================================
A vibrant school of iridescent koi fish glide gracefully through a crystal-clear azure sky, their shimmering scales catching golden sunlight that filters through billowing cumulus clouds. Below, an emerald valley dotted with wildflowers gazes upward at this surreal spectacle, where turquoise fins ripple like silk ribbons in the warm breeze, creating a dreamlike fusion of ocean and atmosphere.
```
生成美化后的GIF
```
python inference.py \
  --prompt="A vibrant school of iridescent koi fish glide gracefully through a crystal-clear azure sky, their shimmering scales catching golden sunlight that filters through billowing cumulus clouds. Below, an emerald valley dotted with wildflowers gazes upward at this surreal spectacle, where turquoise fins ripple like silk ribbons in the warm breeze, creating a dreamlike fusion of ocean and atmosphere." \
  --seed 452 --precision f32 \
  --output="fish_new.gif"
```

![horse_new](https://github.com/user-attachments/assets/bb6df2a8-f357-44e3-a7be-6ab1c9693e7a)


注：文件夹中提供了三组GIF优化对比；具体代码执行过程及终端日志见code.ipynb文件


## 参考文献
[1] https://aistudio.baidu.com/projectdetail/8321341

[2] https://api-docs.deepseek.com/zh-cn/

