# Qwen2-VL 模型服务器

基于 FastAPI 的 Qwen2 VL（视觉语言）模型服务。

## 文件结构

- `server.py`: FastAPI 主服务程序
- `start_server.sh`: 服务启动脚本（支持参数配置）

## 环境要求

- Python 3.8+
- PaddlePaddle develop(0.0.0)【commit号: f41f081861203441adf3f235bfa854c6fd312d1d】
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

POST `/generate`

请求体格式：
```json
{
  "messages": [...],
  "max_new_tokens": 128(默认)
}
```

返回格式：
```json
{
  "error_code": 200,
  "error_msg": "success",
  "result": {
    "response": {
      "role": "assistant",
      "utterance": "生成的文本"
    }
  }
}
```

参考调用脚本：
```python
import requests

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "paddlemix/demo_images/examples_image1.jpg",
                },
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ],
    "max_new_tokens": 128
}

response = requests.post("http://0.0.0.0:8001/generate", json=data)
print(response.json())
# {'error_code': 200, 'error_msg': 'success', 'result': {'response': {'role': 'assistant', 'utterance': 'A red panda is sitting on a wooden box.'}}}
```
