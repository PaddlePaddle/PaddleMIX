# PaddleMIX 扩展插件 for ComfyUI

## 简介
[ComfyUI](https://github.com/comfyanonymous/ComfyUI/) 是一个在开源社区广受欢迎的AIGC程序。它通过节点拆分和工作流组合的方式，让不同模型协同工作，完成复杂的高级生产任务。本目录包含PaddleMIX为ComfyUI开发的一些节点扩展程序，支持文本到图像生成、图像分割、图像生成文本描述等多模态能力。

## 安装与使用指南

### 一、准备ComfyUI环境

#### 从源代码部署
访问 [ComfyUI GitHub仓库](https://github.com/comfyanonymous/ComfyUI) 获取源代码。

#### 使用Docker部署
1. **拉取镜像文件压缩包并加载**（或直接使用 `docker pull` 命令拉取网上的任意ComfyUI镜像）：
    ```shell
    wget https://paddlenlp.bj.bcebos.com/models/community/aistudio/comfyui_docker/comfyui_aistudio_v1.tar
    docker load -i comfyui_aistudio_v1.tar
    ```
2. **创建Docker实例**，注意替换路径和镜像名称：
    ```shell
    nvidia-docker run --name comfyui_env -it -e HOME="/root" -w "/root" -v </path/to/temp_data_dir>:/root --ipc=host --net=host <docker-image-name> /bin/bash --login
    ```
3. **进入Docker环境**：
    ```shell
    docker exec -it comfyui_env /bin/bash
    ```
4. **启动ComfyUI**：
    ```shell
    cd /comfyui_env
    ./python_env/bin/python ComfyUI/main.py --listen 0.0.0.0 --port 8889 &
    ```

### 二、安装PaddleMIX ComfyUI扩展程序

将PaddleMIX/comfyui/下的对应插件文件夹复制到ComfyUI/custom_nodes/文件夹下，并安装对应的requirements.txt文件即可使用。

#### 安装文生图扩展节点的示例：
```shell
# 复制扩展程序文件夹到ComfyUI/custom_nodes/目录
cp -r PaddleMIX/comfyui/ComfyUI_ppdiffusers /path/to/your/ComfyUI/custom_nodes/
# 安装扩展程序所需要的依赖包
pip install -r PaddleMIX/comfyui/ComfyUI_ppdiffusers/requirements.txt
```

### 三、加载工作流

每个扩展程序目录下都有一个workflows文件夹，你可以通过浏览器加载其中的json文件来使用对应的工作流。具体用例可参考：[PaddleMIX ComfyUI扩展程序示例](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/comfyui/ComfyUI_ppdiffusers)。
