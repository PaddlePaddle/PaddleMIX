

### 开放世界检测分割（Grounded-SAM: Detect and Segment Everything with Text Prompt）

`Grounded-SAM` 示例:

```python
from paddlemix.appflow import Appflow
from PIL import Image
task = Appflow(app="openset_det_sam",
               models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"]
               )
image_pil = Image.open("beauty.png").convert("RGB")
result = task(image=image_pil,prompt="women")
```

效果展示

<div align="center">

| Text prompt | Input Image | Generated Mask |
|:----:|:----:|:----:|
| horse,grasses,sky | ![horse](https://github.com/LokeZhou/PaddleMIX/assets/13300429/cae06f3c-a0e3-46cb-8231-6e9eae58bc2b) | ![horse_mask](https://github.com/LokeZhou/PaddleMIX/assets/13300429/3e5e14b9-1089-43d5-8775-1fe678f104b1) |
</div>
