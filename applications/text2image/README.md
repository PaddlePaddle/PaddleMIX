

### 文图生成（Text-to-Image Generation）


```python
import paddle
from paddlemix.appflow import Appflow

paddle.seed(42)
task = Appflow(app="text2image_generation",
               models=["stabilityai/stable-diffusion-xl-base-1.0"]
               )
prompt = "a photo of an astronaut riding a horse on mars."
result = task(prompt=prompt)['result']
```

效果展示

<div align="center">

| model| prompt | Generated Image |
|:----:|:----:|:----:|
|stabilityai/stable-diffusion-v1-5| a photo of an astronaut riding a horse on mars | ![astronaut_rides_horse_sd](https://github.com/LokeZhou/PaddleMIX/assets/13300429/1622fb1e-c841-4531-ad39-9c5092a2456c)|
|stabilityai/stable-diffusion-xl-base-1.0| a photo of an astronaut riding a horse on mars |![sdxl_text2image](https://github.com/LokeZhou/PaddleMIX/assets/13300429/9e339d97-18cd-4cfc-89a6-c545e2872f7e) |
</div>
