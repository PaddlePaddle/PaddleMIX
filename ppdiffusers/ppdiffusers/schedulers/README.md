# 调度器

PPDiffusers提供了许多用于扩散过程的调度器函数。调度器接收模型的输出（扩散过程正在迭代的样本）和一个时间步长，返回去噪后的样本。时间步长很重要，因为它决定了步骤在扩散过程中的位置；数据通过向前迭代*n*个时间步长生成，推断通过向后传播时间步长进行。根据时间步长，调度器可以是*离散*的，此时时间步长为`int`，或者*连续*的，此时时间步长为`float`。

根据上下文，调度器定义了如何迭代地向图像添加噪声或者如何根据预训练模型的输出更新样本：

- 在*训练*期间，调度器向样本添加噪声（有不同的算法可用于添加噪声），以训练扩散模型。
- 在*推断*期间，调度器定义了如何根据预训练模型的输出更新样本。

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
- `use_auth_token`（str或bool，可选） - 用作远程文件的HTTP Bearer授权的令牌。如果为`True`，则使用从diffusers-cli login生成的令牌（存储在~/.huggingface中）。
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

- `prev_sample`（torch.FloatTensor，形状为(batch_size, num_channels, height, width)） - 上一个时间步的计算样本(x_{t-1})。`prev_sample`应该在去噪循环中作为下一个模型输入使用。

`SchedulerOutput`是调度器`step`函数输出的基类，用于存储和传递给下一个时间步的信息。它是一个抽象类，可以由具体的调度器子类继承和实现。

## KarrasDiffusionSchedulers

[`KarrasDiffusionSchedulers`]是PPDiffusers中调度器的广义概括。该类中的调度器在高层次上通过其噪声采样策略、网络和缩放类型、训练策略以及损失加权方式进行区分。

根据常微分方程（ODE）求解器类型的不同，该类中的不同调度器属于上述分类，并为PPDiffusers中实现的主要调度器的设计提供了良好的抽象。该类中的调度器在[这里](https://github.com/huggingface/diffusers/blob/a69754bb879ed55b9b6dc9dd0b3cf4fa4124c765/src/diffusers/schedulers/scheduling_utils.py#L32)给出。
