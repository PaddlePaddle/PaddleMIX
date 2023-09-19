

### 开放世界检测分割（Grounded-SAM: Detect and Segment Everything with Text Prompt）

`Grounded-SAM` 示例:

```python
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

task = Appflow(task="openset_det_sam",
                   models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"],
                   static_mode=False) #如果开启静态图推理，设置为True,默认动态图
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
image_pil = load_image(url)
result = task(image=image_pil,prompt="dog")

plt.figure(figsize=(10, 10))
plt.imshow(image_pil)
for mask in result['seg_masks']:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)


plt.axis('off')
plt.savefig(
    'dog.jpg',
    bbox_inches="tight", dpi=300, pad_inches=0.0
)

```

效果展示

<div align="center">

| Text prompt | Input Image | Generated Mask |
|:----:|:----:|:----:|
| dog | ![overture-creations](https://github.com/LokeZhou/PaddleMIX/assets/13300429/fe13b5f6-e773-41c2-9660-3b2747575fc1) | ![dog](https://github.com/LokeZhou/PaddleMIX/assets/13300429/f472cbd9-7b68-4699-888c-d4ea87fa8256) |
| horse,grasses,sky | ![horse](https://github.com/LokeZhou/PaddleMIX/assets/13300429/cae06f3c-a0e3-46cb-8231-6e9eae58bc2b) | ![horse_mask](https://github.com/LokeZhou/PaddleMIX/assets/13300429/3e5e14b9-1089-43d5-8775-1fe678f104b1) |
</div>


