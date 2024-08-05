# PaddleInfer Stable Video Diffusion 模型高性能部署

 **目录**
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)
       * [基于图像条件的视频生成（Image-to-Video Generation）](#基于图像条件的视频生成)

⚡️[PaddleInfer]是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 PaddleInfer 将我们 PPDiffusers 训练好的 Stable Diffusion XL模型进行多硬件、多推理引擎后端高性能部署。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 PaddleInfer，需要执行以下命令安装依赖。

```shell
python -m pip install paddlepaddle-gpu==2.6.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

<a name="快速体验"></a>

## 静态图模型导出 (static model export)
```
export USE_PPXFORMERS=False
python export_model.py --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt --output_path static_model/stable-video-diffusion-img2vid-xt
```
导出模型在static_model/stable-video-diffusion-img2vid-xt目录下。

### 基于图像条件的视频生成（Image-to-Video Generation）
```
python infer.py --model_dir static_model/stable-video-diffusion-img2vid-xt --scheduler "ddim" --backend paddle --width 256 --height 256 --device gpu --task_name img2video
```
