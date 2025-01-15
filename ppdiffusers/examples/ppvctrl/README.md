# PP-VCtrl
PP-VCtrl 是一个通用的视频生成控制模型，通过引入辅助条件编码器，能够灵活对接各类控制模块，并且在不改变原始生成器的前提下避免了大规模重训练。该模型利用稀疏残差连接实现对控制信号的高效传递，同时通过统一的条件编码流程，将多种控制输入转换为标准化表示，再结合任务特定掩码以提升适应性。得益于这种统一而灵活的设计，PP-VCtrl 可广泛应用于人物动画、场景转换、视频编辑等视频生成场景。

<img src="assets/models/model.png" style="width:100%">




<!-- **[PP-Vctrl: Controlable Video Generation Models](https://arxiv.org/absadada/)** 
</br> -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2307-b31b1b.svg)](https://arxiv.org/abs/) -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://https://github.com/PaddlePaddle/PaddleMIX.github.io/) -->
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/) -->

<!-- ***注意:***  -->
## 📰 新闻
 `[2025-01-08]`:🔥模型推理代码，推理脚本，模型权重，已经发布。训练代码敬请期待！
## 🚩 **TODO/最新进展**
- [x] PP-VCtrl v1 模型权重
- [x] Inference code


## 📷 快速展示
### 1. 边缘控制的视频生成 (Canny)：
<table class="center">
    <thead>
        <tr>
            <th>Text</th> <!-- 新增的列标题，在最左边 -->
            <th>Reference</th>
            <th>Control Videos</th>
            <th>Ours T2V</th>
            <th>Ours I2V</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Group of fishes swimming in aquarium.</td> <!-- 新增的文本描述，在最左边 -->
            <td><img src="assets/figures/canny_case1_reference.jpg" alt="Reference " width="160"></td>
            <td><img src="assets/figures/canny_case1_control_image.gif" alt="Conrotl Videos" width="160"></td>
            <td><img src="assets/figures/canny_case1_ours_t2v.gif" alt="Ours T2V" width="160"></td>
            <td><img src="assets/figures/canny_case1_ours_i2v.gif" alt="Ours I2V" width="160"></td>
        </tr>
        <tr>
            <td>A boat with a flag on it is sailing on the sea.</td> <!-- 第二行的文本描述 -->
            <td><img src="assets/figures/canny_case2_reference.jpg" alt="Reference" width="160"></td>
            <td><img src="assets/figures/canny_case2_control_image.gif" alt="Control Videos" width="160"></td>
            <td><img src="assets/figures/canny_case2_ours_t2v.gif" alt="Ours T2v" width="160"></td>
            <td><img src="assets/figures/canny_case2_ours_i2v.gif" alt="Ours I2v" width="160"></td>
        </tr>
        <!-- 可以继续添加更多行 -->
    </tbody>
</table>


### 2. 蒙版控制的视频生成 (Mask)：
<table class="center">
    <thead>
        <tr>
            <th>Text</th> <!-- 新增的列标题，在最左边 -->
            <th>Reference</th>
            <th>Control Videos</th>
            <th>Ours T2V</th>
            <th>Ours I2V</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>A rider in a dark helmet and white breeches is atop a chestnut horse...</td> <!-- 新增的文本描述，在最左边 -->
            <td><img src="assets/figures/mask_case1_reference.jpg" alt="Reference " width="160"></td>
            <td><img src="assets/figures/mask_case1_control_image.gif" alt="Conrotl Videos" width="160"></td>
            <td><img src="assets/figures/mask_case1_ours_t2v.gif" alt="Ours T2V" width="160"></td>
            <td><img src="assets/figures/mask_case1_ours_i2v.gif" alt="Ours I2V" width="160"></td>
        </tr>
        <tr>
            <td>A dark gray Mini Cooper is parked on a city street...</td> <!-- 第二行的文本描述 -->
            <td><img src="assets/figures/mask_case2_reference.jpg" alt="Reference" width="160"></td>
            <td><img src="assets/figures/mask_case2_control_image.gif" alt="Control Videos" width="160"></td>
            <td><img src="assets/figures/mask_case2_ours_t2v.gif" alt="Ours T2v" width="160"></td>
            <td><img src="assets/figures/mask_case2_ours_i2v.gif" alt="Ours I2v" width="160"></td>
        </tr>
        <!-- 可以继续添加更多行 -->
    </tbody>
</table>

### 3. 人体姿态图控制的视频生成 (Pose)：
<table class="center">
    <thead>
        <tr>
            <th>Text</th> <!-- 新增的列标题，在最左边 -->
            <th>Reference</th> <!-- 新增的列标题，在最左边 -->
            <th>Pose Videos</th>
            <th>Ours</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>A young man with curly hair and a red t-shirt featuring a white logo is seen in various states of motion... </td>  
            <td><img src="assets/figures/pose_case1_reference1.jpg" alt="Reference 1" width="160"></td> 
           <td><img src="assets/figures/pose_case1_control_image.gif" alt="Pose Videos" width="160"></td>
            <td><img src="assets/figures/pose_case1_ours_1.gif" alt="Ours 1" width="160"></td>
        </tr>
        <tr>
            <td>A woman models an Adrianna Papell women's gown, featuring a sleeveless...</td> 
            <td><img src="assets/figures/pose_case2_reference2.jpg" alt="Reference 1" width="160"></td> 
            <td><img src="assets/figures/pose_case2_control_image.gif" alt="Pose Videos" width="160"></td>
            <td><img src="assets/figures/pose_case2_ours_2.gif" alt="Ours 1" width="160"></td>
        </tr>
        <!-- 可以继续添加更多行 -->
    </tbody>
</table>
    



## 🚀 快速开始
***注意:*** 
PP-Vctrl模型是建立在 PaddlePaddle 和 ppdiffusers 上的。以下是使用和操作说明。

### 1. 设置仓库和环境
```bash

# 创建python环境
conda create -n PP-VCtrl python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate PP-VCtrl
```
```bash
# 安装paddlenlp
pip install paddlenlp==v3.0.0-beta2
# 安装3.0.0-beta-2版本的paddlepaddle-gpu，当前我们选择了cuda11.8的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

```
```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX.git
```
```bash
# 安装ppdiffusers
cd PaddleMIX
pip install -e ppdiffusers
# 进入 vctrl目录
cd ppdiffusers/examples/vctrl
# 安装其他所需的依赖
pip install -r requirements.txt
```
### 2. 下载预训练权重
请按照如下方式下载权重：

#### 2.1 下载PP-VCtrl-canny模型权重
```bash
#创建vctrl-canny模型权重目录
mkdir -p weights/canny

#下载i2v模型权重
wget -P weights/canny https://bj.bcebos.com/v1/dataset/PaddleMIX/vctrl/paddle_weights/vctrl_canny_5b_i2v_vctrl-tiny.pdparams

#下载t2v模型权重
wget -P weights/canny https://bj.bcebos.com/v1/dataset/PaddleMIX/vctrl/paddle_weights/vctrl_canny_5b_t2v.pdparams

```
#### 2.2 下载PP-VCtrl-msak模型权重
```bash
#创建vctrl-mask模型权重目录
mkdir -p weights/mask

#下载i2v模型权重
wget -P weights/mask https://bj.bcebos.com/v1/dataset/PaddleMIX/vctrl/paddle_weights/vctrl_5b_i2v_mask.pdparams

#下载t2v模型权重
wget -P weights/mask https://bj.bcebos.com/v1/dataset/PaddleMIX/vctrl/paddle_weights/vctrl_5b_t2v_mask.pdparams


```
#### 2.3 下载PP-VCtrl-pose模型权重
```bash
#创建vctrl-poses模型权重目录
mkdir -p weights/poses

#下载i2v模型权重
wget -P weights/poses https://bj.bcebos.com/v1/dataset/PaddleMIX/vctrl/paddle_weights/vctrl_pose_5b_i2v.pdparams
```

### 3. 准备预测试数据
所有测试数据按照以下规则放置，我们已经为你提供了所需的测试案例：
```
examples/
├── pose/case-1
│   ├── guide_values.mp4
│   ├── prompt.txt
│   └── reference_image.jpg
├── mask/case-1
│   ├── guide_values.mp4
|   ├── mask_values.mp4
│   ├── prompt.txt
│   └── reference_image.jpg
├── canny/case-1
│   ├── guide_values.mp4
│   ├── prompt.txt
│   └── reference_image.jpg
```


## 🔥 模型推理和视频生成
模型的最终推理结果可以在 **/infer_outputs** 中找到。
#### 3.1 通过边缘控制生成视频
```bash
##t2v
bash scripts/infer_cogvideox_i2v_canny_vctrl.sh

##i2v
bash scripts/infer_cogvideox_t2v_canny_vctrl.sh
```

#### 3.2 通过蒙版控制生成视频
```bash
##t2v
bash scripts/infer_cogvideox_t2v_mask_vctrl.sh

##i2v
bash scripts/infer_cogvideox_i2v_mask_vctrl.sh
```
***注意:*** 边缘和蒙版控制模型可以同时支持t2v和i2v模型。 
#### 3.3 通过人物姿态图控制生成视频
```bash
##i2v
bash scripts/infer_cogvideox_i2v_pose_vctrl.sh
```
***注意:*** 人物姿态控制模型只适用于i2v模型。 
#### 3.3 Gradio 应用
我们还创建了一个 Gradio 应用，供您与我们的模型进行交互。
<!-- ```
```
<img src="asserts/figs/gradio.jpg" style="width:70%"> -->


## 📚 技术细节



### 4.1 PP-VCtrl
在当今数字创意领域，视频生成技术已成为内容创作和叙事表达的重要工具。近期文本到视频的扩散模型虽然实现了自然语言驱动的视频生成，但在控制生成内容的精细时空特征方面仍面临重大挑战。 比如，在在广告创意、影视后期制作、直播带货、虚拟人交互等应用场景下，仅依靠文本接口难以精确指定物体轮廓、人体姿态以及画面背景等视觉特征，这些都需要更精确的控制信号来引导生成过程。目前的创作者往往需要通过反复调整文本描述来接近预期效果，这种试错式的迭代不仅耗时低效，也难以完全满足视频生成中对精确控制的需求，亟需更有效的视频控制方案。

尽管ControlNet在可控图像生成领域取得了突破性进展，但视频生成领域仍缺乏类似的通用控制方案。当前可控视频生成的研究主要集中在开发特定任务的解决方案，如人物动画生成、视频修复和运动控制等。这些方法通常为每个具体任务设计专门的模块，导致技术体系碎片化，缺乏统一的理论框架。同时，它们在处理文本提示和参考帧等基础输入时往往受限于任务特定的设计，难以实现灵活的跨任务迁移。此外，现有的一些方法试图通过控制图像生成模型来生成视频，而不是直接控制视频生成模型，这在时序一致性和整体生成质量上都存在局限。

针对上述挑战，我们提出了PP-VCtrl：一个统一的视频生成控制框架，它通过引入辅助条件编码器，实现了对各类控制信号的灵活接入和精确控制，同时保持了高效的计算性能。它可以高效地应用在各类视频生成场景，尤其是在人物动画、场景转换、视频编辑等需要精确控制的任务中。

### 4.2 数据策略
相比于文本/图像-视频生成，可控视频生成的数据除了满足画面质量、文本-视频对齐外，还需要根据不同的可控任务构造不同的数据集。我们通过收集公开视频数据集构建原始数据池，对原始数据进行切分单镜头、去除黑边、水印和字幕后，进行美学质量评分过滤得到可用数据池。基于可用数据池做recaption、人体关节点提取和视频分割，依次满足canny、pose和mask视频编辑任务的数据需求。具体如下图所示：
<img src="assets/models/data1.png" style="width:100%">

通过上述的流程，我们得到了可控视频生成数据集在不同任务的训练数据分布：
<img src="assets/models/data2.png" style="width:100%">

### 4.3 训练策略
为了提升模型的泛化能力和鲁棒性，我们采用了多样化的数据增强和训练策略。在去噪过程中，通过正弦函数采样时间步，以更好地关注视频生成的关键阶段。在空间维度上，默认情况下采用基于正态分布的裁剪策略，根据视频宽高比自适应地进行裁剪，在增强数据多样性的同时也能使模型很好地关注视频主体内容。

针对不同任务特点，我们设计了相应的优化策略。在边缘控制任务中，采用动态阈值采样增加数据多样性；对于人体姿态控制任务，针对横竖版视频分别采用填充和裁剪的预处理策略；在蒙版控制任务中，我们采用基于区域面积权重的多目标采样方法，根据概率分布动态选择目标区域，并支持区域扩展和多目标联合控制，同时通过随机概率的膨胀处理来增强模型鲁棒性，使生成结果更加自然。这些策略在统一的视频生成控制框架基础上进行综合运用，显著提升了模型在各类场景下的适应能力和生成质量，并充分发挥了PP-VCtrl通用控制框架的优势。
### 4.4 定量指标评测
在边缘控制视频生成（Canny）、人体姿态控制视频生成（Pose）以及蒙版控制视频生成（Mask）三个任务的定量评估中，PPVCtrl模型在控制能力和视频质量指标上均能够媲美或超越现有开源的特定任务方法。

<img src="assets/models/eval1.png" style="width:100%">

我们进行了人工评估实验，邀请了多位评估者对不同方法生成的视频进行打分，评估维度包括视频整体质量、时序一致性等。结果显示，在所有评估维度上，PPVCtrl的评分均高于现有开源方法。

<img src="assets/models/eval2.png" style="width:100%">
</details>
<!-- 
## More version
<details close>
<summary>Model Versions</summary>
</details>
-->
<!-- 
## Contact us
Users: [Users@example.com](Users@example.com)  
-->
<!-- 
 ## BibTex

```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Liang, Zhengyang and Wang, Yaohui and Qiao, Yu and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={International Conference on Learning Representations},
  year={2025}
}

```上面的代码打印了一条消息 -->