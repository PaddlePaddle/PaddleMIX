# Qwen2-VL 模型服务器

基于 FastAPI 的 Qwen2 VL（视觉语言）模型服务。

## 文件结构

- `server.py`: FastAPI 主服务程序
- `start_server.sh`: 服务启动脚本（支持参数配置）

## 环境要求

- Python 3.8+
- PaddlePaddle 3.0.0b2或develop(0.0.0)【commit号: f41f081861203441adf3f235bfa854c6fd312d1d】
- PaddleNLP（3.0.0b2)

## 配置参数

| 参数 | 默认值 | 说明 |
|-----|--------|-----|
| host | 0.0.0.0 | 服务器绑定地址 |
| port | 8001 | 服务器端口号 |
| model-path | Qwen/Qwen2-VL-2B-Instruct | 模型权重路径 |

## 快速开始

1. 使用默认配置启动：
```bash
sh paddlemix/examples/qwen2_vl/start_server.sh 
```

2. 自定义配置启动：
```bash
sh paddlemix/examples/qwen2_vl/start_server.sh  --host 127.0.0.1 --port 8080 --model-path /自定义/模型/路径
```

## API 接口

POST `/v1/chat/completions`

```

参考调用脚本：
```python
import base64
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key='xxxxxxxxx',
    base_url='http://10.67.188.11:8080/v1/chat/completions' 
)


#图片转base64函数
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#输入图片路径
image_path = "paddlemix/demo_images/examples_image1.jpg"
 
#原图片转base64
base64_image = encode_image(image_path)

#提交信息至GPT4o
response = client.chat.completions.create(
    model="pp-docbee",#选择模型
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content":[
            {
          "type": "text",
          "text": "Describe this image."
        },
                    {
          "type": "image_url",
          "image_url":{
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        },
        ]
        }
    ],
    stream=True,
)
# 流式响应
reply = ""
for res in response:
    content = res.choices[0].delta.content
    if content:
        reply += content
        print(content)

print('reply:',reply)

# # 非流式响应
# content = response.choices[0].message.content
# print('Reply:', content)
```
