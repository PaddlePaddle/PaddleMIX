

### 文图生成（Text-to-Image Generation）


```python
from paddlemix import Appflow

paddle.seed(1024)
task = Appflow(app="text2image_generation",
               models=["stabilityai/stable-diffusion-v1-5"]
               )
prompt = "a photo of an astronaut riding a horse on mars."
result = task(prompt=prompt)['result']
```

效果展示

<div align="center">

| prompt | Generated Image |
|:----:|:----:|
| a photo of an astronaut riding a horse on mars | ![astronaut_rides_horse_sd](https://github.com/LokeZhou/PaddleMIX/assets/13300429/457ca26c-452e-4692-bf5f-515ea41920d7) |
</div>
