

### 检测框引导的图像编辑（Det-Guided-Inpainting)

`Grounded-SAM-Inpainting` 示例:

```python
from paddlemix import Appflow
from PIL import Image
task = Appflow(app="inpainting",
               models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024","stabilityai/stable-diffusion-2-inpainting"]
               )
paddle.seed(1024)
image_pil = Image.open("bus.png").convert("RGB")
result = task(image=image_pil,prompt="bus",inpaint_prompt="a yellow van")
```
<div align="center">

| Input Image | Det Prompt | Generated Mask | Inpaint Prompt | Inpaint Image |
|:----:|:----:|:----:|:----:|:----:|
| ![bus](https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11) | bus | ![text_inapinting_seg](https://github.com/LokeZhou/PaddleMIX/assets/13300429/5b68fc15-aebe-4e05-b420-edd6989a66ef)| a yellow van | ![text_inpainting](https://github.com/LokeZhou/PaddleMIX/assets/13300429/451da53c-3b7d-4a9d-8063-01a92eae0768)|

</div>


### 文本检测框引导的图像编辑（ChatAndDet-Guided-Inpainting)
`Grounded-SAM-chatglm` 示例:

```python
from paddlemix import Appflow
from PIL import Image
task = Appflow(app="inpainting",
               models=["THUDM/chatglm-6b","GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024","stabilityai/stable-diffusion-2-inpainting"]
               )
paddle.seed(1024)
image_pil = Image.open("bus.png").convert("RGB")
result = task(image=image_pil,inpaint_prompt="bus is changed to A school bus parked on the roadside")
```

一些效果展示

<div align="center">

| Input Image | Prompt | Generated Mask | Inpaint Prompt |
|:----:|:----:|:----:|:----:|
| ![bus](https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11) |  bus is changed to A school bus parked on the roadside | ![chat_inpainting_seg](https://github.com/LokeZhou/PaddleMIX/assets/13300429/dedf9943-6ef2-42df-b4ad-b8336208b283)| ![chat_inpainting](https://github.com/LokeZhou/PaddleMIX/assets/13300429/1e3c2cdb-8202-41ee-acc9-b56e6b53005c)|

</div>
