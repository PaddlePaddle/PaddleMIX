# Audio Chat

## 1. 应用简介

Enter audio and prompt words for question and answer.

*****
- No training is need.
- Integration with the moedel of 🤗  [whisper](), [chatglm](). [fastspeech2]().

----

## 2. Demo
*****
example:

```python
#audio_chat 
from paddlemix import Appflow
import paddle
paddle.seed(1024)
task = Appflow(app="audio_chat", models=["whisper", "THUDM/chatglm-6b", "speech"])
audio_file = "./zh.wav"
prompt = (
    "描述这段话：{}."
)
output_path = "tmp.wav"
result = task(audio=audio_file, prompt=prompt, output=output_path)

# 这段话表达了作者认为跑步最重要的好处之一是身体健康。作者认为,通过跑步,身体得到了良好的锻炼,身体健康得到了改善。作者还强调了跑步对身体健康的重要性,并认为这是最值得投资的运动之一。

```

|  输入音频 | 输入prompt | 输出文本 | 输出结果 |
| --- | --- | ---  | --- | 
|[zh.wav](https://github.com/luyao-cv/file_download/blob/main/assets/zh.wav) | "描述这段话." |"这段话表达了作者认为跑步最重要的好处之一是身体健康。作者认为,通过跑步,身体得到了良好的锻炼,身体健康得到了改善。作者还强调了跑步对身体健康的重要性,并认为这是最值得投资的运动之一。" |[audiochat-result.wav](https://github.com/luyao-cv/file_download/blob/main/assets/zh.wav)|
