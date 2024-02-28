# NaViT
Paddle Variation of [Patch n’ Pack: NaViT, a Vision Transformer for all Aspect Ratios and Resolutions](https://arxiv.org/pdf/2307.06304.pdf)

# Usage
```python
import paddle
from navit import NaViT

v = NaViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    token_dropout_prob = 0.1
)
v.eval()

images = [
    [paddle.randn([3, 256, 256]), paddle.randn([3, 128, 128])],
    [paddle.randn([3, 128, 256]), paddle.randn([3, 256, 128])],
    [paddle.randn([3, 64, 256])]
]
preds = v(images)

print(preds)
```

# Appreciation
* Lucidrains
* Agorians


# Citations
```
@misc{2307.06304,
Author = {Mostafa Dehghani and Basil Mustafa and Josip Djolonga and Jonathan Heek and Matthias Minderer and Mathilde Caron and Andreas Steiner and Joan Puigcerver and Robert Geirhos and Ibrahim Alabdulmohsin and Avital Oliver and Piotr Padlewski and Alexey Gritsenko and Mario Lučić and Neil Houlsby},
Title = {Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution},
Year = {2023},
Eprint = {arXiv:2307.06304},
}
```
