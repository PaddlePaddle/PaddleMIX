基于Hotshot-XL与DeepSeek模型的文生图优化

在原有基础上实现生成图片的内容优化

第一行的图片为原始图片，第二行的图片为优化后的图片
![gradio](https://github.com/user-attachments/assets/30dbc1bd-a344-4d94-b014-59b183feedd6)

将原始文本基于DeepSeek-R1进行拓展完善：
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

```
!python deepseek.py "Fish are flying in the sky."

Okay, the user wants me to beautify the description "Fish are flying in the sky." for a text-to-image prompt. Let me think about how to enhance this. First, I need to add more vivid details. Maybe specify the type of fish, like tropical or koi. Then, the sky – perhaps a clear azure sky with fluffy clouds. Lighting is important; maybe golden sunlight to make the colors pop. The fish could have iridescent scales shimmering in the light. Adding movement elements, like schools of fish gliding gracefully. The atmosphere should be surreal and magical. Maybe mention the environment below, like a serene meadow or mountains to contrast with the flying fish. Also, using words like "ethereal" and "dreamlike" to set the mood. Let me put it all together cohesively.
=================================================
A vibrant school of iridescent koi fish glide gracefully through a crystal-clear azure sky, their shimmering scales catching golden sunlight that filters through billowing cumulus clouds. Below, an emerald valley dotted with wildflowers gazes upward at this surreal spectacle, where turquoise fins ripple like silk ribbons in the warm breeze, creating a dreamlike fusion of ocean and atmosphere.
```

```
!python inference.py \
  --prompt="A vibrant school of iridescent koi fish glide gracefully through a crystal-clear azure sky, their shimmering scales catching golden sunlight that filters through billowing cumulus clouds. Below, an emerald valley dotted with wildflowers gazes upward at this surreal spectacle, where turquoise fins ripple like silk ribbons in the warm breeze, creating a dreamlike fusion of ocean and atmosphere." \
  --seed 452 --precision f32 \
  --output="fish_new.gif"
```
