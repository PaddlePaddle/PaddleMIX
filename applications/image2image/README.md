### 文本引导的图像放大（Text-Guided Image Upscaling

```python
from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"

low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"

app = Appflow(app='image2image_text_guided_upscaling',models=['stabilityai/stable-diffusion-x4-upscaler'])
image = app(prompt=prompt,image=low_res_img)['result']

image.save("upscaled_white_cat.png")
```

效果展示

<div align="center">

| prompt |image | Generated Image |
|:----:|:----:|:----:|
| a white cat| ![low_res_cat](https://github.com/LokeZhou/PaddleMIX/assets/13300429/5cc5f2ee-5709-4722-b5f2-3adabe98cbf2) |![upscaled_white_cat](https://github.com/LokeZhou/PaddleMIX/assets/13300429/f5688dd6-b328-4c3f-a9ab-9575b6ee77b2) |
</div>




### 文本图像双引导图像生成（Dual Text and Image Guided Generation）

```python
from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)
prompt = "a red car in the sun"


app = Appflow(app='dual_text_and_image_guided_generation',models=['shi-labs/versatile-diffusion'])
image = app(prompt=prompt,image=image)['result']
image.save("versatile-diffusion-red_car.png")

```

效果展示

<div align="center">

| prompt |image | Generated Image |
|:----:|:----:|:----:|
| a red car in the sun | ![benz](https://github.com/LokeZhou/PaddleMIX/assets/13300429/2a71f5fd-3dd3-4f3b-a3cb-fe5282eb728b) | ![versatile-diffusion-red_car](https://github.com/LokeZhou/PaddleMIX/assets/13300429/3904d53e-5412-4896-92d0-43c5770d8b39)|
</div>



### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

```python
from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image
import paddle

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
image = load_image(url).resize((512, 768))
prompt = "a red car in the sun"

paddle.seed(42)
prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"


app = Appflow(app='image2image_text_guided_generation',models=['admruul/anything-v3.0'])
image = app(prompt=prompt,negative_prompt=negative_prompt,image=image)['result']

image.save("image_Kurisu_img2img.png")

```

效果展示

<div align="center">

| prompt | negative_prompt |image | Generated Image |
|:----:|:----:|:----:| :----:|
| Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress | lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry | ![image_Kurisu](https://github.com/LokeZhou/PaddleMIX/assets/13300429/9596c6b9-2dea-4a66-9419-b60332a08cd1)|![image_Kurisu_img2img](https://github.com/LokeZhou/PaddleMIX/assets/13300429/f4fa0efe-bce2-4bea-b6f6-19591af7e423) |
</div>
