# Music Generation

## 1. 应用简介

Enter audio and prompt words for question and answer.

*****
- No training is need.
- Integration with the moedel of 🤗  [minigpt4](), [minigpt4](), [chatglm]().

----

## 2. Demo
*****
example:

```python
#music generation
from paddlemix import Appflow
import paddle
from PIL import Image
import scipy
paddle.seed(1024)

# Text to music
task = Appflow(app="music_generation", models=["cvssp/audioldm"])
prompt = "A classic cocktail lounge vibe with smooth jazz piano and a cool, relaxed atmosphere."
negative_prompt = "low quality, average quality"
num_inference_steps = 20
audio_length_in_s = 10
output_path = "tmp.wav"
result = task(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s, generator = paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)

# image to music
task1 = Appflow(app="music_generation", models=["miniGPT4/MiniGPT4-7B"])
negative_prompt = "low quality, average quality"
num_inference_steps = 20
audio_length_in_s = 10
output_path = "tmp.wav"
minigpt4_text = 'describe the image, '
image_pil = Image.open("tmp.jpg").convert("RGB")
result = task1(image=image_pil, minigpt4_text=minigpt4_text, )['result'].split('#')[0]
paddle.device.cuda.empty_cache()
# miniGPT4 output: The image shows a pineapple cocktail sitting on a table in front of a person. The pineapple is cut in half and the drink is poured into the top half. The person is holding a straw in their hand and appears to be sipping the drink. There are also some other items on the table, such as a plate with food and a glass of water. The background is a marble table with a pattern on it.
prompt = "Given the scene description in the following paragraph, please create a musical style sentence that fits the scene.Description:{}.".format(result)
task2 = Appflow(app="music_generation", models=["THUDM/chatglm-6b", "cvssp/audioldm"])
result = task2(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s, generator = paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)
# chatglm ouptput: The music swells as the image shows the pineapple cocktail on the table, with the drink cut in half and the person sipping it with a straw. The background is a marble table with a pattern, and the other items on the table are a plate with food and a glass of water. The music fades until it disappears, leaving the scene in the person's hand the pineapple drink, with the music once again swelling in the background.
```

