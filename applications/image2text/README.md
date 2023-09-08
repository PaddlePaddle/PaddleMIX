

### 图文生成（Image-to-Text Generation）

## miniGPT4
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
from paddlemix.appflow import Appflow
import requests

task = Appflow(app="image2text_generation",
               models=["miniGPT4/MiniGPT4-7B"])
url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
image = Image.open(requests.get(url, stream=True).raw)
minigpt4_text = "describe the image"
result = task(image=image,minigpt4_text=minigpt4_text)
```

效果展示

<div align="center">

| Image | text | Generated text|
|:----:|:----:|:----:|
|![mugs](https://github.com/LokeZhou/PaddleMIX/assets/13300429/b5a95002-bb30-4683-8e62-ed21879f24e1) | describe the image|The image shows two mugs with cats on them, one is black and white and the other is blue and white. The mugs are sitting on a table with a book in the background. The mugs have a whimsical, cartoon-like appearance. The cats on the mugs are looking at each other with a playful expression. The overall style of the image is cute and fun.###|
</div>

## blip2

```python
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image

task = Appflow(app="image2text_generation",
               models=["paddlemix/blip2-caption-opt2.7b"])
url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
image_pil = load_image(url)
blip2_prompt = 'describe the image'
result = task(image=image_pil,blip2_prompt=blip2_prompt)
```

| Image | text | Generated text|
|:----:|:----:|:----:|
|![mugs](https://github.com/LokeZhou/PaddleMIX/assets/13300429/b5a95002-bb30-4683-8e62-ed21879f24e1) | describe the image|of the two coffee mugs with cats on them|
</div>
