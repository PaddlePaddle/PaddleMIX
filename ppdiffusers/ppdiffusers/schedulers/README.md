# 调度器

PPDiffusers提供了许多用于扩散过程的调度器函数。调度器接收模型的输出（扩散过程正在迭代的样本）和一个时间步长，返回去噪后的样本。时间步长很重要，因为它决定了步骤在扩散过程中的位置；数据通过向前迭代*n*个时间步长生成，推理通过向后传播时间步长进行。根据时间步长，调度器可以是*离散*的，此时时间步长为`int`，或者*连续*的，此时时间步长为`float`。

根据上下文，调度器定义了如何迭代地向图像添加噪声或者如何根据预训练模型的输出更新样本：

- 在*训练*期间，调度器向样本添加噪声（有不同的算法可用于添加噪声），以训练扩散模型。
- 在*推理*期间，调度器定义了如何根据预训练模型的输出更新样本。

许多调度器是由[Katherine Crowson](https://github.com/crowsonkb/)的[k-diffusion](https://github.com/crowsonkb/k-diffusion)库实现的，并且在A1111中也广泛使用。为了帮助您将k-diffusion和A1111中的调度器映射到PPDiffusers中的调度器，请查看下对照表：

| A1111/k-diffusion    | PPDiffusers                         | 用法                                                                                                          |
|---------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| DPM++ 2M            | [`DPMSolverMultistepScheduler`]     |                                                                                                               |
| DPM++ 2M Karras     | [`DPMSolverMultistepScheduler`]     | 使用`use_karras_sigmas=True`进行初始化                                                                            |
| DPM++ 2M SDE        | [`DPMSolverMultistepScheduler`]     | 使用`algorithm_type="sde-dpmsolver++"`进行初始化                                                                  |
| DPM++ 2M SDE Karras | [`DPMSolverMultistepScheduler`]     | 使用`use_karras_sigmas=True`和`algorithm_type="sde-dpmsolver++"`进行初始化                                     |
| DPM++ 2S a          | N/A                                 | 与`DPMSolverSinglestepScheduler`非常相似                         |
| DPM++ 2S a Karras   | N/A                                 | 与`DPMSolverSinglestepScheduler(use_karras_sigmas=True, ...)`非常相似 |
| DPM++ SDE           | [`DPMSolverSinglestepScheduler`]    |                                                                                                               |
| DPM++ SDE Karras    | [`DPMSolverSinglestepScheduler`]    | 使用`use_karras_sigmas=True`进行初始化                                                                            |
| DPM2                | [`KDPM2DiscreteScheduler`]          |                                                                                                               |
| DPM2 Karras         | [`KDPM2DiscreteScheduler`]          | 使用`use_karras_sigmas=True`进行初始化                                                                            |
| DPM2 a              | [`KDPM2AncestralDiscreteScheduler`] |                                                                                                               |
| DPM2 a Karras       | [`KDPM2AncestralDiscreteScheduler`] | 使用`use_karras_sigmas=True`进行初始化                                                                            |
| DPM自适应            | N/A                                 |                                                                                                               |
| DPM快速             | N/A                                 |                                                                                                               |
| Euler               | [`EulerDiscreteScheduler`]          |                                                                                                               |
| Euler a             | [`EulerAncestralDiscreteScheduler`] |                                                                                                               |
| Heun                | [`HeunDiscreteScheduler`]           |                                                                                                               |
| LMS                 | [`LMSDiscreteScheduler`]            |                                                                                                               |
| LMS Karras          | [`LMSDiscreteScheduler`]            | 使用`use_karras_sigmas=True`进行初始化                                                                            |
| N/A                 | [`DEISMultistepScheduler`]          |                                                                                                               |
| N/A                 | [`UniPCMultistepScheduler`]         |                                                                                                               |

所有调度器都是从基类[`SchedulerMixin`]构建的，该基类实现了所有调度器共享的低级方法。

## SchedulerMixin

`SchedulerMixin`是所有调度器的基类。

`SchedulerMixin`包含了所有调度器共享的通用函数，例如通用的加载和保存功能。

类属性：

- `_compatibles`（List[str]） - 一个包含与父调度器类兼容的调度器类的列表。可以使用`from_config()`方法加载不同的兼容调度器类（应由父类覆盖）。

### 方法

#### from_pretrained

```python
from_pretrained(pretrained_model_name_or_path: Union[str, os.PathLike, None] = None, subfolder: Optional[str] = None, return_unused_kwargs: bool = False, **kwargs)
```

参数：

- `pretrained_model_name_or_path`（str或os.PathLike，可选） - 可以是以下中的一个：
  - 字符串，预训练模型在Hub上的模型ID（例如：google/ddpm-celebahq-256）。
  - 目录路径，包含使用`save_pretrained()`保存的调度器配置。
- `subfolder`（str，可选） - Hub上或本地存储库中模型文件的子文件夹位置。
- `return_unused_kwargs`（bool，可选，默认为`False`） - 是否返回未被Python类使用的kwargs。
- `cache_dir`（Union[str, os.PathLike]，可选） - 预训练模型配置的缓存目录路径（如果不使用标准缓存）。
- `force_download`（bool，可选，默认为`False`） - 是否强制（重新）下载模型权重和配置文件，覆盖缓存版本（如果存在）。
- `resume_download`（bool，可选，默认为`False`） - 是否恢复下载模型权重和配置文件。如果设置为`False`，则删除任何未完全下载的文件。
- `proxies`（Dict[str, str]，可选） - 代理服务器字典，用于每个协议或端点，例如`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`。代理服务器将在每个请求上使用。
- `output_loading_info`（bool，可选，默认为`False`） - 是否同时返回一个包含丢失的键、意外的键和错误消息的字典。
- `local_files_only`（bool，可选，默认为`False`） - 是否仅加载本地模型权重和配置文件。如果设置为`True`，则不会从Hub下载模型。
- `revision`（str，可选，默认为`"main"`） - 要使用的特定模型版本。可以是分支名称、标签名称、提交ID或Git允许的任何标识符。

#### save_pretrained

```python
save_pretrained(save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs)
```

参数：

- `save_directory`（str或os.PathLike） - 保存配置JSON文件的目录（如果不存在，将创建该目录）。
- `push_to_hub`（bool，可选，默认为`False`） - 是否在保存后将模型推送到Hugging Face Hub。您可以使用`repo_id`指定要推送到的存储库（默认为命名空间中`save_directory`的名称）。
- `kwargs`（Dict[str, Any]，可选） - 传递给`push_to_hub()`方法的其他关键字参数。

将调度器配置对象保存到目录中，以便可以使用`from_pretrained()`类方法重新加载。

## SchedulerOutput

`SchedulerOutput`是调度器`step`函数的输出的基类。

### 参数

- `prev_sample`（paddle.Tensor，形状为(batch_size, num_channels, height, width)） - 上一个时间步的计算样本(x_{t-1})。`prev_sample`应该在去噪循环中作为下一个模型输入使用。

`SchedulerOutput`是调度器`step`函数输出的基类，用于存储和传递给下一个时间步的信息。它是一个抽象类，可以由具体的调度器子类继承和实现。

## KarrasDiffusionSchedulers

[`KarrasDiffusionSchedulers`]是PPDiffusers中调度器的广义概括。该类中的调度器在高层次上通过其噪声采样策略、网络和缩放类型、训练策略以及损失加权方式进行区分。

根据常微分方程（ODE）求解器类型的不同，该类中的不同调度器属于上述分类，并为PPDiffusers中实现的主要调度器的设计提供了良好的抽象。该类中的调度器在[这里](https://github.com/huggingface/diffusers/blob/a69754bb879ed55b9b6dc9dd0b3cf4fa4124c765/src/diffusers/schedulers/scheduling_utils.py#L32)给出。

## CMStochasticIterativeScheduler

Consistency Models是由Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever所提出的一种多步和一步调度器（算法1），能够在一步或少量步骤中生成优质样本。

论文摘要如下：

扩散模型在图像、音频和视频生成方面取得了重大突破，但它们依赖于迭代生成过程，导致采样速度慢，限制了其在实时应用中的潜力。为了克服这个限制，我们提出了一致性模型，这是一种新的生成模型系列，可以在不进行对抗训练的情况下实现高质量样本。它们通过设计支持快速的一步生成，同时仍然允许少量步骤的采样以在计算和样本质量之间进行权衡。它们还支持零样本数据编辑，如图像修复、上色和超分辨率，而无需对这些任务进行显式训练。一致性模型可以作为提炼预训练扩散模型的一种方法进行训练，也可以作为独立的生成模型进行训练。通过大量实验证明，它们在一步和少量步骤生成方面优于现有的扩散模型提炼技术。例如，在CIFAR-10上实现了新的FID最佳成绩3.55，在ImageNet 64x64上实现了6.20的FID最佳成绩。当作为独立的生成模型进行训练时，一致性模型还在CIFAR-10、ImageNet 64x64和LSUN 256x256等标准基准测试中优于单步非对抗生成模型。

原始代码可在[openai/consistency_models](https://github.com/openai/consistency_models)找到。

## ConsistencyDecoderScheduler

这个调度器是ConsistencyDecoderPipeline的一部分，在[DALL-E 3](https://openai.com/dall-e-3)中引入。

原始代码可在[openai/consistency_models](https://github.com/openai/consistency_models)找到。

## DDIMInverseScheduler

`DDIMInverseScheduler`是根据Jiaming Song、Chenlin Meng和Stefano Ermon的[ Denoising Diffusion Implicit Models （DDIM）](https://huggingface.co/papers/2010.02502)中的倒置调度器而设计的。该实现主要基于使用引导扩散模型进行真实图像编辑的[Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://huggingface.co/papers/2211.09794.pdf)。

## DDIMScheduler

Denoising Diffusion Implicit Models (DDIM) by Jiaming Song, Chenlin Meng, and Stefano Ermon

本文摘要如下：

Denoising diffusion probabilistic models (DDPMs)在没有对抗训练的情况下实现了高质量图像生成，但是它们需要模拟多个步骤的马尔可夫链才能生成样本。为了加速采样过程，我们提出了denoising diffusion implicit models (DDIMs)，这是一种更高效的迭代隐式概率模型类，其训练过程与DDPMs相同。在DDPMs中，生成过程被定义为马尔可夫扩散过程的逆过程。我们构建了一类非马尔可夫扩散过程，其导致相同的训练目标，但其逆过程的采样速度要快得多。我们经验证实，与DDPMs相比，DDIMs在墙钟时间上可以以10倍至50倍的速度生成高质量的样本，允许我们在计算和样本质量之间进行权衡，并且可以直接在潜空间中执行语义有意义的图像插值。

该论文的原始代码可以在ermongroup/ddim找到，您可以在tsong.me上联系作者。

提示：
论文Common Diffusion Noise Schedules and Sample Steps are Flawed声称训练和推理设置之间的不匹配导致了Stable Diffusion的推理生成结果不佳。为了解决这个问题，作者提出了以下方法：

🧪 这是一个实验性的功能！

重新调整噪声计划以强制实现零终端信噪比（SNR）
```python
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True)
```
使用v_prediction训练模型（将以下参数添加到train_text_to_image.py或train_text_to_image_lora.py脚本中）
```python
--prediction_type="v_prediction"
```
将采样器始终从最后一个时间步开始
```python
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
```
重新调整无分类器引导以防止过曝光
```python
image = pipeline(prompt, guidance_rescale=0.7).images[0]
```
例如：
```python
from ppdiffusers import DiffusionPipeline, DDIMScheduler

pipe = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", paddle_dtype=paddle.float16)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)

prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"
image = pipeline(prompt, guidance_rescale=0.7).images[0]
```

## DDPMScheduler

[Denoising Diffusion Probabilistic Models（DDPM）](https://huggingface.co/papers/2006.11239)是由Jonathan Ho、Ajay Jain和Pieter Abbeel提出的一种基于扩散的模型。在PPDiffusers库中，DDPM指的是论文中的离散去噪调度器以及整个流程。

论文的摘要如下：

我们使用扩散概率模型进行高质量图像合成，这是一类受非平衡热力学考虑启发的潜变量模型。我们通过根据扩散概率模型与Langevin动力学的去噪分数匹配之间的新连接设计的加权变分界限进行训练，获得了最佳结果。我们的模型自然地采用了一种渐进的有损解压缩方案，可以解释为自回归解码的一般化。在无条件的CIFAR10数据集上，我们获得了9.46的Inception分数和3.17的最先进的FID分数。在256x256的LSUN数据集上，我们获得了与ProgressiveGAN相似的样本质量。

## DEISMultistepScheduler

[Fast Sampling of Diffusion Models with Exponential Integrator ](https://huggingface.co/papers/2204.13902)一文中，Qinsheng Zhang和Yongxin Chen提出了DEIS（Diffusion Exponential Integrator Sampler）。DEISMultistepScheduler是扩散常微分方程（ODE）的一种快速高阶求解器。

这个实现修改了DEIS论文中原始线性t空间的多项式拟合公式，在对数-密度空间中进行了修改。这种修改利用了指数多步更新的封闭形式系数，而不是依赖于数值求解器。

摘要如下：

过去几年中，扩散模型（DMs）在生成式建模任务中生成高保真样本取得了巨大成功。DM的一个主要限制是其臭名昭著的缓慢采样过程，通常需要数百到数千个时间离散步骤才能达到所需的精度。我们的目标是开发一种快速的DM采样方法，只需较少的步骤即可保持高样本质量。为此，我们系统地分析了DM中的采样过程，并确定了影响样本质量的关键因素，其中离散化方法最为关键。通过仔细研究学习到的扩散过程，我们提出了Diffusion Exponential Integrator Sampler（DEIS）。它基于为离散化常微分方程（ODEs）设计的指数积分器，并利用了学习到的扩散过程的半线性结构来减小离散化误差。该方法可应用于任何DM，并可以在仅需10个步骤时生成高保真样本。在我们的实验中，使用一个A6000 GPU从CIFAR10生成5万张图像大约需要3分钟。此外，通过直接使用预训练的DMs，在限制评分函数评估次数（NFE）的情况下，我们在CIFAR10上实现了最先进的采样性能，例如在10个NFE的情况下，FID为4.17，而在15个NFE的情况下，FID为3.37，IS为9.74。代码可在此[https URL](https://github.com/qsh-zh/deis)找到。

原始代码库可在[qsh-zh/deis](https://github.com/qsh-zh/deis)找到。

提示：
建议将solver_order设置为2或3，而solver_order=1相当于DDIMScheduler。

支持来自Imagen的动态阈值设定，对于基于像素空间的扩散模型，可以设置thresholding=True来使用动态阈值。

## DPMSolverMultistepInverse

**DPMSolverMultistepInverse** 是来自 [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) 和 [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095)，作者是 Cheng Lu、Yuhao Zhou、Fan Bao、Jianfei Chen、Chongxuan Li 和 Jun Zhu。

该实现主要基于 DDIM 反演定义的 [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://huggingface.co/papers/2211.09794.pdf) 以及 [Xiang-cd/DiffEdit-stable-diffusion](https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/diffedit.ipynb) 的 DiffEdit 潜空间反演的笔记本实现。

提示
支持来自 Imagen 的动态阈值化（https://huggingface.co/papers/2205.11487 ），对于像素空间的扩散模型，您可以将 algorithm_type="dpmsolver++" 和 thresholding=True 同时设置为使用动态阈值化。这种阈值化方法不适用于稳定扩散等潜空间扩散模型。

## DPMSolverMultistepScheduler

DPMSolverMultistep是[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927)，可以在大约10步内完成采样。[DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095)是其改进版本，用于引导扩散概率模型的快速求解。

DPMSolver（和改进版本DPMSolver++）是一种快速的专用高阶求解器，用于扩散ODE，具有收敛阶保证。经验证明，仅使用20步的DPMSolver采样即可生成高质量的样本，即使在10步内也可以生成相当好的样本。

提示
建议在引导采样时将solver_order设置为2，无条件采样时设置为solver_order=3。

支持来自Imagen的动态阈值设置（https://huggingface.co/papers/2205.11487 ），对于像素空间的扩散模型，您可以同时设置algorithm_type="dpmsolver++"和thresholding=True来使用动态阈值设置。这种阈值设置方法不适用于稳定扩散等潜在空间扩散模型。

DPMSolver和DPM-Solver++也支持SDE变体，但仅适用于一阶和二阶求解器。这是一种用于反向扩散SDE的快速求解器。建议使用二阶sde-dpmsolver++。

## DPMSolverSDEScheduler

DPMSolverSDEScheduler是受到[Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)论文中的随机采样器的启发，调度器是由[Katherine Crowson](https://github.com/crowsonkb/)移植和创建的。

## DPMSolverSinglestepScheduler

DPMSolverSinglestepScheduler 是来自 [ DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps ](https://huggingface.co/papers/2206.00927)，大约在10步内完成采样，以及 [ DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models ](https://huggingface.co/papers/2211.01095)，作者为 Cheng Lu、Yuhao Zhou、Fan Bao、Jianfei Chen、Chongxuan Li 和 Jun Zhu。

DPMSolver（以及改进版本 DPMSolver++）是一个快速的专用高阶求解器，用于扩散常微分方程，具有收敛阶数保证。经验上，DPMSolver 仅需20步采样即可生成高质量样本，即使在10步内也能生成相当好的样本。

原始实现可在 [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver) 找到。

提示
建议在引导采样中将 solver_order 设置为2，而在无条件采样中将 solver_order 设置为3。

支持来自 Imagen 的动态阈值方法（https://huggingface.co/papers/2205.11487 ），对于像素空间扩散模型，您可以设置 algorithm_type="dpmsolver++" 和 thresholding=True 来使用动态阈值方法。这种阈值方法不适用于稳定扩散等潜在空间扩散模型。

## EulerAncestralDiscreteScheduler

一个使用祖先采样和欧拉方法步骤的调度器。这是一个快速调度器，通常可以在20-30步内生成良好的输出。该调度器基于Katherine Crowson的原始k-扩散实现。

## EulerDiscreteScheduler

Euler调度器（算法2）来自Karras等人的[ Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)论文。这是一个快速调度器，通常可以在20-30步内生成良好的输出。该调度器基于Katherine Crowson的原始k-diffusion实现。

## HeunDiscreteScheduler

Heun调度器（算法1）来自Karras等人的[ Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)论文。该调度器是由Katherine Crowson从k-diffusion库移植而来的。

## IPNDMScheduler

IPNDMScheduler是一个四阶改进的伪线性多步调度器。原始实现可以在[crowsonkb/v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch/blob/987f8985e38208345c1959b0ea767a625831cc9b/diffusion/sampling.py#L296)找到。

## KarrasVeScheduler

KarrasVeScheduler是一个针对方差扩展（VE）模型的随机采样器。它基于[Elucidating the Design Space of Diffusion-Based Generative Models ](https://huggingface.co/papers/2206.00364)和[Score-based generative modeling through stochastic differential equations](https://huggingface.co/papers/2011.13456)这两篇论文。

## KDPM2AncestralDiscreteScheduler

`KDPM2DiscreteScheduler`是受到[Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)一文的启发，并且采用了祖先抽样方法。该调度程序是由Katherine Crowson移植和创建的。

原始代码库可以在[crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion)找到。

## KDPM2DiscreteScheduler

`KDPM2DiscreteScheduler`是受到[Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)论文的启发而创建的，调度器是由Katherine Crowson进行移植和创建的。

原始代码库可以在[crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion)找到。

## Latent Consistency Model Multistep Scheduler

在[ Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)一文中，Simian Luo、Yiqin Tan、Longbo Huang、Jian Li和Hang Zhao引入了多步和单步调度器（算法3），并与潜在一致性模型一起使用。该调度器能够在1-8步内从LatentConsistencyModelPipeline生成良好的样本。

## LMSDiscreteScheduler

LMSDiscreteScheduler 是一个用于离散beta调度的线性多步调度器。该调度器是由 Katherine Crowson 移植和创建的，原始实现可以在 [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181) 找到。

## PNDMScheduler

PNDMScheduler（伪数值方法调度器）是一种用于扩散模型的高级常微分方程（ODE）积分技术，包括Runge-Kutta和线性多步法。你可以在[crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181)找到原始实现。

## RePaintScheduler

RePaintScheduler是一种基于DDPM的自动修复调度器，用于无监督修复具有极端遮罩的图像。它设计用于与RePaintPipeline配合使用，并基于Andreas Lugmayr等人的论文[ RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://huggingface.co/papers/2201.09865)。

论文摘要如下：

自由形式的修复是在由任意二进制遮罩指定的区域向图像添加新内容的任务。大多数现有方法针对特定遮罩分布进行训练，这限制了它们对未见过的遮罩类型的泛化能力。此外，使用像素级和感知损失进行训练通常会导致对缺失区域的简单纹理扩展，而不是语义上有意义的生成。在这项工作中，我们提出了RePaint：一种基于去噪扩散概率模型（DDPM）的修复方法，适用于极端遮罩。我们使用预训练的无条件DDPM作为生成先验。为了对生成过程进行条件化，我们仅通过使用给定图像信息对未遮罩区域进行采样来修改反向扩散迭代。由于这种技术不修改或条件化原始的DDPM网络本身，该模型可以为任何修复形式生成高质量和多样化的输出图像。我们使用标准和极端遮罩验证了我们的方法，包括人脸和通用图像修复。RePaint在六个遮罩分布中至少有五个超过了最先进的自回归和GAN方法。Github仓库：git.io/RePaint。

可以在[andreas128/RePaint](https://github.com/andreas128/)找到原始实现。

## ScoreSdeVeScheduler

ScoreSdeVeScheduler 是一个方差爆炸的随机微分方程调度器。它是由Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole在[ Score-Based Generative Modeling through Stochastic Differential Equations ](https://huggingface.co/papers/2011.13456)论文中提出的。

论文摘要如下：

从数据中创建噪声很容易，通过噪声创建数据则是生成建模。我们提出了一个随机微分方程（SDE），通过缓慢注入噪声，将复杂的数据分布平滑地转化为已知的先验分布，并提出了一个相应的逆向时间 SDE，通过缓慢消除噪声将先验分布转化回数据分布。关键是，逆向时间 SDE 仅依赖于扰动数据分布的时间相关梯度场（也称为评分）。通过利用基于评分的生成建模的进展，我们可以使用神经网络准确地估计这些评分，并使用数值 SDE 求解器生成样本。我们展示了这个框架包含了以前的基于评分的生成建模和扩散概率建模方法，从而允许新的采样过程和新的建模能力。特别地，我们引入了一个预测校正框架来纠正离散化逆向时间 SDE 演化中的错误。我们还推导出了一个等价的神经常微分方程（ODE），它从与 SDE 相同的分布中进行采样，但还能实现精确的似然计算和改进的采样效率。此外，我们提供了一种使用基于评分模型解决逆问题的新方法，通过在类别条件生成、图像修复和着色等实验中进行了验证。结合多个架构改进，我们在 CIFAR-10 上实现了无条件图像生成的创纪录性性能，Inception 分数为 9.89，FID 为 2.20，似然度为 2.99 bits/dim，并首次展示了基于评分的生成模型对 1024 x 1024 图像的高保真生成能力。

## ScoreSdeVpScheduler

ScoreSdeVpScheduler是一个保持方差的随机微分方程（SDE）调度器。它在Yang Song、Jascha Sohl-Dickstein、Diederik P. Kingma、Abhishek Kumar、Stefano Ermon、Ben Poole的论文[ Score-Based Generative Modeling through Stochastic Differential Equations ](https://huggingface.co/papers/2011.13456)中首次提出。

论文摘要如下：

从数据中创建噪声很容易，而从噪声中创建数据则是生成建模。我们提出了一种随机微分方程（SDE），通过缓慢注入噪声，将复杂的数据分布平滑地转化为已知的先验分布，并且提出了相应的逆时SDE，通过缓慢去除噪声将先验分布转化回数据分布。关键是，逆时SDE仅依赖于扰动数据分布的时间相关梯度场（也称为评分）。通过利用基于评分的生成建模的进展，我们可以使用神经网络准确地估计这些评分，并使用数值SDE求解器生成样本。我们展示了这个框架包含了以前在基于评分的生成建模和扩散概率建模中的方法，从而允许新的采样过程和新的建模能力。特别地，我们引入了一个预测-校正框架来纠正离散逆时SDE演化中的错误。我们还推导出了一个等效的神经ODE，它从与SDE相同的分布中采样，但还能够进行精确的似然计算和改进的采样效率。此外，我们提供了一种使用基于评分模型解决逆问题的新方法，通过在类条件生成、图像修复和上色等实验中进行了验证。结合多种架构改进，我们在CIFAR-10上实现了无条件图像生成的创纪录性性能，Inception分数为9.89，FID为2.20，似然度为2.99 bits/dim，并首次展示了从基于评分的生成模型中生成1024 x 1024图像的高保真度。

## UniPCMultistepScheduler

UniPCMultistepScheduler是一个无需训练的框架，旨在快速采样扩散模型。它是由Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu在[UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://huggingface.co/papers/2302.04867) 中引入的。

该框架由一个校正器（UniC）和一个预测器（UniP）组成，它们共享统一的分析形式并支持任意阶数。UniPC是为任何模型设计的，支持像素空间/潜空间DPMs的无条件/条件采样。它还可以应用于噪声预测模型和数据预测模型。校正器UniC也可以应用于任何现成的求解器之后，以提高准确度。

论文摘要如下：

扩散概率模型（DPMs）在高分辨率图像合成方面展示出非常有前景的能力。然而，从预训练的DPM中采样通常需要数百次模型评估，计算成本很高。尽管近年来在设计DPM的高阶求解器方面取得了进展，但仍有进一步加速的空间，特别是在非常少的步骤（例如5~10步）中。受ODE求解器的预测校正方法的启发，我们开发了一个统一的校正器（UniC），可以在任何现有的DPM采样器之后应用，以提高准确度，而无需额外的模型评估，并推导出一个支持任意阶数的统一预测器（UniP）作为副产品。结合UniP和UniC，我们提出了一种统一的预测校正框架，称为UniPC，用于快速采样DPMs，它对任何阶数都具有统一的分析形式，并且可以显著提高采样质量。我们通过广泛的实验评估了我们的方法，包括使用像素空间和潜空间DPMs的无条件和条件采样。我们的UniPC在CIFAR10（无条件）上可以达到3.87 FID，在ImageNet 256x256（条件）上可以达到7.51 FID，仅需10次函数评估。代码可在https://github.com/wl-zhao/UniPC 找到。

原始代码库可在[wl-zhao/UniPC](https://github.com/wl-zhao/UniPC) 中找到。

提示：
对于引导采样，建议将solver_order设置为2，对于无条件采样，将solver_order设置为3。

支持来自Imagen（https://huggingface.co/papers/2205.11487 ）的动态阈值设置，对于像素空间扩散模型，您可以将predict_x0和thresholding都设置为True以使用动态阈值。对于稳定扩散等潜空间扩散模型，此阈值方法不适用。


## VQDiffusionScheduler

VQDiffusionScheduler是将转换器模型的输出转化为上一个扩散时间步长中无噪声图像的样本。它是由Shuyang Gu、Dong Chen、Jianmin Bao、Fang Wen、Bo Zhang、Dongdong Chen、Lu Yuan和Baining Guo在[ Vector Quantized Diffusion Model for Text-to-Image Synthesis ](https://huggingface.co/papers/2111.14822)一文中引入的。

该论文的摘要如下：

我们提出了一种基于向量量化扩散（VQ-Diffusion）模型的文本到图像生成方法。该方法基于一个向量量化变分自编码器（VQ-VAE），其潜空间由最近开发的去噪扩散概率模型（DDPM）的条件变体建模。我们发现，这种潜空间方法非常适合文本到图像生成任务，因为它不仅消除了现有方法中的单向偏差，还允许我们采用掩码替换扩散策略来避免误差的积累，这是现有方法中的一个严重问题。我们的实验表明，与具有相似参数数量的传统自回归（AR）模型相比，VQ-Diffusion在文本到图像生成结果方面表现出显著优势。与之前基于GAN的文本到图像方法相比，我们的VQ-Diffusion可以处理更复杂的场景，并显著提高合成图像的质量。最后，我们展示了我们方法中的图像生成计算可以通过重新参数化实现高效。传统的自回归方法中，文本到图像生成时间与输出图像分辨率呈线性增长，因此即使对于普通大小的图像，也需要相当长的时间。VQ-Diffusion允许我们在质量和速度之间取得更好的平衡。我们的实验表明，带有重新参数化的VQ-Diffusion模型比传统的自回归方法快15倍，同时实现更好的图像质量。
