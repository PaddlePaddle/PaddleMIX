# PPdiffusers WebUI

![](./static/image_hutao.png)

1.克隆仓库

```bash
git clone https://github.com/PaddlePaddle/PaddleMIX.git
cd PaddleMIX/ppdiffusers/examples/StableDiffusionXLWebUI
```

2.克隆模型仓库，获取所需的模型权重（权重没有在bce模型库中，需要从自定义模型空间转载）

```bash
git lfs install
git clone http://git.aistudio.baidu.com/2510368/Paddle-Stable-Diffusion.git
```

复制运行所需的模型权重

```bash
cp -r Paddle-Stable-Diffusion/ip-adapter .
cp -r Paddle-Stable-Diffusion/Pony_Pencil-Xl-V1.0.2 .
rm -rf Paddle-Stable-Diffusion
```

3.依赖

```bash
pip install -r requirements.txt --user
```

### Demo

更多内容，在 `main.ipynb`中查看。


### 启动webui

进入`StableDiffusionXLWebUI`目录后，*双击运行* `webui_adapter.gradio.py`，32GB GPU可运行。

或者*双击运行* `webui_ctr.gradio.py`，32GB GPU可运行。

或者*双击运行* `webui.gradio.py`, 16GB GPU可运行。(将权重加载到cpu使用时再加载回gpu，会牺牲速度但是可以换得超分模型的执行，可以通过`Load To CPU`选项关闭。)
