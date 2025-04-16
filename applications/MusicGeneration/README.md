### 音乐生成（Music Generation）

#### 1. Application introduction

Enter audio and prompt words for question and answer.

*****
- No training is need.
- Integration with the model of [minigpt4](), [chatglm](), [audioldm]().

----

#### 2. Demo
*****
example:


使用miniGPT4前，需要下载相应权重进行转换，具体可参考[miniGPT4](../../paddlemix/examples/minigpt4/README.md),在完成权重转换后，根据模型权重文件以及配置文件按下存放：
```bash
--PPMIX_HOME  #默认路径 /root/.paddlemix  可通过export PPMIX_HOME 设置
  --models
    --miniGPT4
      --MiniGPT4-7B
        config.json
        model_state.pdparams
        special_tokens_map.json
        image_preprocessor_config.json
        preprocessor_config.json
        tokenizer_config.json
        model_config.json
        sentencepiece.bpe.model
        tokenizer.json
      --MiniGPT4-13B
        ...
        ...
    ...

```
完成之后，可使用appflow 一键预测

```python
#music generation
from paddlemix.appflow import Appflow
import paddle
from PIL import Image
import scipy
paddle.seed(1024)

# Text to music
task = Appflow(app="music_generation", models=["cvssp/audioldm"])
prompt = "A classic cocktail lounge vibe with smooth jazz piano and a cool, relaxed atmosphere."
negative_prompt = 'low quality, average quality, muffled quality, noise interference, poor and low-grade quality, inaudible quality, low-fidelity quality'  
audio_length_in_s = 5
num_inference_steps = 20
output_path = "tmp.wav"
result = task(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s, generator = paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)

# image to music
task1 = Appflow(app="music_generation", models=["miniGPT4/MiniGPT4-7B"])
negative_prompt = 'low quality, average quality, muffled quality, noise interference, poor and low-grade quality, inaudible quality, low-fidelity quality'  
audio_length_in_s = 5
num_inference_steps = 20
output_path = "tmp.wav"
minigpt4_text = 'describe the image, '
image_pil = Image.open("dance.png").convert("RGB")
result = task1(image=image_pil, minigpt4_text=minigpt4_text )['result'].split('#')[0]
paddle.device.cuda.empty_cache()
# miniGPT4 output: The image shows a crowded nightclub with people dancing on the dance floor. The lights on the dance floor are green and red, and there are several people on the dance floor. The stage is at the back of the room, and there are several people on stage. The walls of the nightclub are decorated with neon lights and there are several people sitting at tables in the background. The atmosphere is lively and energetic.

prompt = "Given the scene description in the following paragraph, please create a musical style sentence that fits the scene.  Description:{}.".format(result)
task2 = Appflow(app="music_generation", models=["THUDM/chatglm-6b", "cvssp/audioldm"])
result = task2(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s, generator = paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)
# chatglm output: The music is playing, and the crowd is dancing like never before. The lights are bright and the atmosphere is electric, with people swaying to the rhythm of the music and the energy of the night. The dance floor is a sea of movement, with people moving to the music and feeling the rhythm of their feet. The stage is a place of magic, with people on it, performing their best. The neon lights of the nightclub are a testament to the energy and excitement of the night, with people's faces lit up as they perform. And as the music continues to play, the crowd continues to dance, never letting up, until the night is over. 
```


#### Text to music
|  Input Prompt | Output Music |
| --- | --- |
|'A classic cocktail lounge vibe with smooth jazz piano and a cool, relaxed atmosphere.'| [jazz_output.wav](https://github.com/luyao-cv/file_download/blob/main/assets/jazz_output.wav)

---

#### image to music
|  Input Image | Output Caption | Output Text | Output Music |
| --- | --- |  --- |  --- | 
|![dance.png](https://github.com/luyao-cv/file_download/blob/main/vis_music_generation/dance.png) | 'The image shows a crowded nightclub with people dancing on the dance floor. The lights on the dance floor are green and red, and there are several people on the dance floor. The stage is at the back of the room, and there are several people on stage. The walls of the nightclub are decorated with neon lights and there are several people sitting at tables in the background. The atmosphere is lively and energetic.' | 'The music is playing, and the crowd is dancing like never before. The lights are bright and the atmosphere is electric, with people swaying to the rhythm of the music and the energy of the night. The dance floor is a sea of movement, with people moving to the music and feeling the rhythm of their feet. The stage is a place of magic, with people on it, performing their best. The neon lights of the nightclub are a testament to the energy and excitement of the night, with people's faces lit up as they perform. And as the music continues to play, the crowd continues to dance, never letting up, until the night is over.' | [dance_output.wav](https://github.com/luyao-cv/file_download/blob/main/assets/dance_output.wav)