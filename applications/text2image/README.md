

### 文图生成（Text-to-Image Generation）


```python
from paddlemix import Appflow
from PIL import Image

paddle.seed(1024)
task = Appflow(app="text2image_generation",
               models=["stabilityai/stable-diffusion-2"]
               )
prompt = "a photo of an astronaut riding a horse on mars."
result = task(prompt=prompt)['result']
```

效果展示

<div align="center">

| prompt | Generated Image |
|:----:|:----:|
| a photo of an astronaut riding a horse on mars |  |
</div>
