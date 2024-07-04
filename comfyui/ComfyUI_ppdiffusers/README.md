# ComfyUI_ppdiffusers
这是一个针对ComfyUI开发的 [ppdiffusers](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers)扩展，支持了常见的模型加载转换、文生图、图生图、图像局部修改等任务。
## 安装
```bash
git clone https://github.com/PaddlePaddle/PaddleMIX --depth 1
cd PaddleMIX/comfyui/
mv ComfyUI_ppdiffusers/ /path/to/your_comfyui/custom_nodes/
```

## 使用
### 在线体验
aistudio: https://aistudio.baidu.com/community/app/106043

### 本地运行
所有已经支持的工作流都在./workflows目录下，可以直接加载工作流json文件使用。
原生支持加载safetensors格式模型，在线读取转换为paddle模型后，在ComfyUI中使用ppdiffusers的pipeline运行。

## 已支持Node
Stable Diffusion 1.5系列：
- SD1.5模型加载转换
- SD1.5文生图
- SD1.5图生图
- SD1.5图像局部修改

Stable Diffusion XL系列：
- SDXL模型加载转换
- SDXL文生图
- SDXL图生图
- SDXL图像局部修改

## 效果展示
### SDXL
1. 文生图  
<img width="600" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/1d9d17cd-dd1f-4e05-9c98-c1fc4fca8185">

2. 图生图  
<img width="600" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/04e9cc05-9ce4-4207-88c4-3d076aaebff4">

3. 局部编辑  
<img width="600" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/36ba7261-1744-41a4-b1cb-c9e99f6931f2">

### SD1.5
1. 文生图  
<img width="600" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/a939274e-a23b-4ecf-956c-56fd8343707c">

2. 图生图  
<img width="600" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/b8b144bb-0b52-4235-91d9-d7bdfb44a1d8">

3. 图像局部重绘  
<img width="600" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/7a75899d-48ca-4479-9fb8-18d077fc3607">
