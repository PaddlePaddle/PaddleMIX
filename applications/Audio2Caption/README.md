### 音频描述（Audio-to-Caption Generation）



#### 1. Application introduction

Enter audio and prompt words for question and answer.

*****
- No training is need.
- Integration with the model of [conformer_u2pp_online_wenetspeech](), [chatglm]().

----

#### 2. Demo
*****
example:

<!-- ```python
python applications/AudioChat/audiochat.py \
--chatglm_question_prompt "please describe this passage." \
--input_audio_file "./zh.wav" \
--chatglm_model_name_or_path "THUDM/chatglm-6b"   \
``` -->
```python
#audio2caption -- Audio to caption converter

from paddlemix.appflow import Appflow
import paddle
paddle.seed(1024)
task = Appflow(app="audio2caption", models=["conformer_u2pp_online_wenetspeech", "THUDM/chatglm-6b"])
audio_file = "./zh.wav"
prompt = (
    "描述这段话：{}."
)
result = task(audio=audio_file, prompt=prompt)['prompt']
print(result)
# 这段话表达了作者认为跑步最重要的好处之一是身体健康。作者认为,通过跑步,身体得到了良好的锻炼,身体健康得到了改善。作者还强调了跑步对身体健康的重要性,并认为这是最值得投资的运动之一。

```

<div align="center">

|  Input Audio | Input Prompt | Output ASR | Output Text |
| --- | --- | ---  | --- | 
|[zh.wav](https://github.com/luyao-cv/file_download/blob/main/assets/zh.wav) | "描述这段话." |"我认为跑步最重要的就是给我带来了身体健康" |这段话表达了作者认为跑步最重要的好处之一是身体健康。作者认为,通过跑步,身体得到了良好的锻炼,身体健康得到了改善。作者还强调了跑步对身体健康的重要性,并认为这是最值得投资的运动之一。 |

<div>

