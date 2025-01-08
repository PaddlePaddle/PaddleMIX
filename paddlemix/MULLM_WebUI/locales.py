# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


LOCALES = {
    "lang": {"en": {"label": "🏷️ Lang"}, "zh": {"label": "🏷️ 语言"}},
    "model_tag": {"zh": "模型", "en": "Model"},
    "image_tag": {"zh": "图像", "en": "Image"},
    "video_tag": {"zh": "视频", "en": "Video"},
    "question_tag": {"zh": "问题", "en": "Question"},
    "model_name": {"en": {"label": "💎 Model name"}, "zh": {"label": "💎 模型名称"}},
    "model_path": {
        "en": {"label": "📁 Model path", "info": "Path to pretrained model or model identifier from Hugging Face."},
        "zh": {"label": "📁 模型路径", "info": "本地模型的文件路径或 Hugging Face 的模型标识符。"},
    },
    "finetuning_type": {"en": {"label": "🔧 Finetuning method"}, "zh": {"label": "🔧 微调方法"}},
    "checkpoint_path": {"en": {"label": "🗂️ Checkpoint path"}, "zh": {"label": "🗂️ 检查点路径"}},
    "template": {
        "en": {"label": "📝 Prompt template", "info": "The template used in constructing prompts."},
        "zh": {"label": "📝 提示模板", "info": "构建提示词时使用的模板。"},
    },
    "rope_scaling": {"en": {"label": "RoPE scaling"}, "zh": {"label": "RoPE 插值方法"}},
    "booster": {"en": {"label": "Booster"}, "zh": {"label": "加速方式"}},
    "training_stage": {
        "en": {"label": "⚙️ Stage", "info": "The stage to perform in training."},
        "zh": {"label": "⚙️ 训练阶段", "info": "目前采用的训练方式。"},
    },
    "dataset_dir": {
        "en": {"label": "💾	Data dir", "info": "Path to the data directory."},
        "zh": {"label": "💾	数据路径", "info": "数据文件夹的路径。"},
    },
    "dataset": {"en": {"label": "📚	Dataset"}, "zh": {"label": "📚 数据集"}},
    "data_preview_btn": {"en": {"value": "🔍 Preview dataset"}, "zh": {"value": "🔍 预览数据集"}},
    "preview_count": {"en": {"label": "Count"}, "zh": {"label": "数量"}},
    "page_index": {"en": {"label": "Page"}, "zh": {"label": "页数"}},
    "prev_btn": {"en": {"value": "Prev"}, "zh": {"value": "上一页"}},
    "next_btn": {"en": {"value": "Next"}, "zh": {"value": "下一页"}},
    "close_btn": {"en": {"value": "Close"}, "zh": {"value": "关闭"}},
    "preview_samples": {"en": {"label": "Samples"}, "zh": {"label": "样例"}},
    "learning_rate": {
        "en": {"label": "⚖️ Learning rate", "info": "Initial learning rate for AdamW."},
        "zh": {"label": "⚖️ 学习率", "info": "AdamW 优化器的初始学习率。"},
    },
    "num_train_epochs": {
        "en": {"label": "⏳ Epochs", "info": "Total number of training epochs to perform."},
        "zh": {"label": "⏳ 训练轮数", "info": "需要执行的训练总轮数。"},
    },
    "max_grad_norm": {
        "en": {"label": "✂️ Maximum gradient norm", "info": "Norm for gradient clipping."},
        "zh": {"label": "✂️ 最大梯度范数", "info": "用于梯度裁剪的范数。"},
    },
    "max_samples": {
        "en": {"label": "🧮 Max samples", "info": "Maximum samples per dataset."},
        "zh": {"label": "🧮 最大样本数", "info": "每个数据集的最大样本数。"},
    },
    "compute_type": {
        "en": {"label": "✨ Compute type", "info": "Whether to use mixed precision training."},
        "zh": {"label": "✨ 计算类型", "info": "是否使用混合精度训练。"},
    },
    "cutoff_len": {
        "en": {"label": "🗜️ Cutoff length", "info": "Max tokens in input sequence."},
        "zh": {"label": "🗜️ 截断长度", "info": "输入序列分词后的最大长度。"},
    },
    "batch_size": {
        "en": {"label": "🧺 Batch size", "info": "Number of samples processed on each GPU."},
        "zh": {"label": "🧺 批处理大小", "info": "每个 GPU 处理的样本数量。"},
    },
    "gradient_accumulation_steps": {
        "en": {"label": "🔗 Gradient accumulation", "info": "Number of steps for gradient accumulation."},
        "zh": {"label": "🔗 梯度累积", "info": "梯度累积的步数。"},
    },
    "val_size": {
        "en": {"label": "💡 Val size", "info": "Proportion of data in the dev set."},
        "zh": {"label": "💡 验证集比例", "info": "验证集占全部样本的百分比。"},
    },
    "lr_scheduler_type": {
        "en": {"label": "🧰 LR scheduler", "info": "Name of the learning rate scheduler."},
        "zh": {"label": "🧰	学习率调节器", "info": "学习率调度器的名称。"},
    },
    "extra_tab": {"en": {"label": "Extra configurations"}, "zh": {"label": "其它参数设置"}},
    "logging_steps": {
        "en": {"label": "Logging steps", "info": "Number of steps between two logs."},
        "zh": {"label": "日志间隔", "info": "每两次日志输出间的更新步数。"},
    },
    "save_steps": {
        "en": {"label": "Save steps", "info": "Number of steps between two checkpoints."},
        "zh": {"label": "保存间隔", "info": "每两次断点保存间的更新步数。"},
    },
    "eval_steps": {
        "en": {"label": "Validation steps", "info": "Number of steps between two evaluations."},
        "zh": {"label": "验证步数间隔", "info": "每两次评估之间的的更新步数。"},
    },
    "warmup_steps": {
        "en": {"label": "Warmup steps", "info": "Number of steps used for warmup."},
        "zh": {"label": "预热步数", "info": "学习率预热采用的步数。"},
    },
    "neftune_alpha": {
        "en": {"label": "NEFTune alpha", "info": "Magnitude of noise adding to embedding vectors."},
        "zh": {"label": "NEFTune 噪声参数", "info": "嵌入向量所添加的噪声大小。"},
    },
    "extra_args": {
        "en": {"label": "Extra arguments", "info": "Extra arguments passed to the trainer in JSON format."},
        "zh": {"label": "额外参数", "info": "以 JSON 格式传递给训练器的额外参数。"},
    },
    "packing": {
        "en": {"label": "Pack sequences", "info": "Pack sequences into samples of fixed length."},
        "zh": {"label": "序列打包", "info": "将序列打包为等长样本。"},
    },
    "neat_packing": {
        "en": {"label": "Use neat packing", "info": "Avoid cross-attention between packed sequences."},
        "zh": {"label": "使用无污染打包", "info": "避免打包后的序列产生交叉注意力。"},
    },
    "train_on_prompt": {
        "en": {"label": "Train on prompt", "info": "Disable the label mask on the prompt (only for SFT)."},
        "zh": {"label": "学习提示词", "info": "不在提示词的部分添加掩码（仅适用于 SFT）。"},
    },
    "mask_history": {
        "en": {"label": "Mask history", "info": "Train on the last turn only (only for SFT)."},
        "zh": {"label": "不学习历史对话", "info": "仅学习最后一轮对话（仅适用于 SFT）。"},
    },
    "resize_vocab": {
        "en": {"label": "Resize token embeddings", "info": "Resize the tokenizer vocab and the embedding layers."},
        "zh": {"label": "更改词表大小", "info": "更改分词器词表和嵌入层的大小。"},
    },
    "use_llama_pro": {
        "en": {"label": "Enable LLaMA Pro", "info": "Make the parameters in the expanded blocks trainable."},
        "zh": {"label": "使用 LLaMA Pro", "info": "仅训练块扩展后的参数。"},
    },
    "shift_attn": {
        "en": {"label": "Enable S^2 Attention", "info": "Use shift short attention proposed by LongLoRA."},
        "zh": {"label": "使用 S^2 Attention", "info": "使用 LongLoRA 提出的 shift short attention。"},
    },
    "report_to": {
        "en": {"label": "Enable external logger", "info": "Use TensorBoard or wandb to log experiment."},
        "zh": {"label": "启用外部记录面板", "info": "使用 TensorBoard 或 wandb 记录实验。"},
    },
    "freeze_tab": {"en": {"label": "Freeze tuning configurations"}, "zh": {"label": "部分参数微调设置"}},
    "freeze_trainable_layers": {
        "en": {
            "label": "Trainable layers",
            "info": "Number of the last(+)/first(-) hidden layers to be set as trainable.",
        },
        "zh": {"label": "可训练层数", "info": "最末尾（+）/最前端（-）可训练隐藏层的数量。"},
    },
    "freeze_trainable_modules": {
        "en": {
            "label": "Trainable modules",
            "info": "Name(s) of trainable modules. Use commas to separate multiple modules.",
        },
        "zh": {"label": "可训练模块", "info": "可训练模块的名称。使用英文逗号分隔多个名称。"},
    },
    "freeze_extra_modules": {
        "en": {
            "label": "Extra modules (optional)",
            "info": "Name(s) of modules apart from hidden layers to be set as trainable. Use commas to separate multiple modules.",
        },
        "zh": {"label": "额外模块（非必填）", "info": "除隐藏层以外的可训练模块名称。使用英文逗号分隔多个名称。"},
    },
    "lora_tab": {"en": {"label": "LoRA configurations"}, "zh": {"label": "LoRA 参数设置"}},
    "lora_rank": {
        "en": {"label": "LoRA rank", "info": "The rank of LoRA matrices."},
        "zh": {"label": "LoRA 秩", "info": "LoRA 矩阵的秩大小。"},
    },
    "lora_alpha": {
        "en": {"label": "LoRA alpha", "info": "Lora scaling coefficient."},
        "zh": {"label": "LoRA 缩放系数", "info": "LoRA 缩放系数大小。"},
    },
    "lora_dropout": {
        "en": {"label": "LoRA dropout", "info": "Dropout ratio of LoRA weights."},
        "zh": {"label": "LoRA 随机丢弃", "info": "LoRA 权重随机丢弃的概率。"},
    },
    "loraplus_lr_ratio": {
        "en": {"label": "LoRA+ LR ratio", "info": "The LR ratio of the B matrices in LoRA."},
        "zh": {"label": "LoRA+ 学习率比例", "info": "LoRA+ 中 B 矩阵的学习率倍数。"},
    },
    "create_new_adapter": {
        "en": {
            "label": "Create new adapter",
            "info": "Create a new adapter with randomly initialized weight upon the existing one.",
        },
        "zh": {"label": "新建适配器", "info": "在现有的适配器上创建一个随机初始化后的新适配器。"},
    },
    "use_rslora": {
        "en": {"label": "Use rslora", "info": "Use the rank stabilization scaling factor for LoRA layer."},
        "zh": {"label": "使用 rslora", "info": "对 LoRA 层使用秩稳定缩放方法。"},
    },
    "use_dora": {
        "en": {"label": "Use DoRA", "info": "Use weight-decomposed LoRA."},
        "zh": {"label": "使用 DoRA", "info": "使用权重分解的 LoRA。"},
    },
    "use_pissa": {
        "en": {"label": "Use PiSSA", "info": "Use PiSSA method."},
        "zh": {"label": "使用 PiSSA", "info": "使用 PiSSA 方法。"},
    },
    "lora_target": {
        "en": {
            "label": "LoRA modules (optional)",
            "info": "Name(s) of modules to apply LoRA. Use commas to separate multiple modules.",
        },
        "zh": {"label": "LoRA 作用模块（非必填）", "info": "应用 LoRA 的模块名称。使用英文逗号分隔多个名称。"},
    },
    "additional_target": {
        "en": {
            "label": "Additional modules (optional)",
            "info": "Name(s) of modules apart from LoRA layers to be set as trainable. Use commas to separate multiple modules.",
        },
        "zh": {"label": "附加模块（非必填）", "info": "除 LoRA 层以外的可训练模块名称。使用英文逗号分隔多个名称。"},
    },
    "rlhf_tab": {"en": {"label": "RLHF configurations"}, "zh": {"label": "RLHF 参数设置"}},
    "pref_beta": {
        "en": {"label": "Beta value", "info": "Value of the beta parameter in the loss."},
        "zh": {"label": "Beta 参数", "info": "损失函数中 beta 超参数大小。"},
    },
    "pref_ftx": {
        "en": {"label": "Ftx gamma", "info": "The weight of SFT loss in the final loss."},
        "zh": {"label": "Ftx gamma", "info": "损失函数中 SFT 损失的权重大小。"},
    },
    "pref_loss": {
        "en": {"label": "Loss type", "info": "The type of the loss function."},
        "zh": {"label": "损失类型", "info": "损失函数的类型。"},
    },
    "reward_model": {
        "en": {"label": "Reward model", "info": "Adapter of the reward model in PPO training."},
        "zh": {"label": "奖励模型", "info": "PPO 训练中奖励模型的适配器路径。"},
    },
    "ppo_score_norm": {
        "en": {"label": "Score norm", "info": "Normalizing scores in PPO training."},
        "zh": {"label": "奖励模型", "info": "PPO 训练中归一化奖励分数。"},
    },
    "ppo_whiten_rewards": {
        "en": {"label": "Whiten rewards", "info": "Whiten the rewards in PPO training."},
        "zh": {"label": "白化奖励", "info": "PPO 训练中将奖励分数做白化处理。"},
    },
    "arg_save_btn": {"en": {"value": "Save arguments"}, "zh": {"value": "保存训练参数"}},
    "arg_load_btn": {"en": {"value": "Load arguments"}, "zh": {"value": "载入训练参数"}},
    "start_btn": {"en": {"value": "Start"}, "zh": {"value": "开始"}},
    "stop_btn": {"en": {"value": "Abort"}, "zh": {"value": "中断"}},
    "output_dir": {
        "en": {"label": "Output dir", "info": "Directory for saving results."},
        "zh": {"label": "输出目录", "info": "保存结果的路径。"},
    },
    "config_path": {
        "en": {"label": "Config path", "info": "Path to config saving arguments."},
        "zh": {"label": "配置路径", "info": "保存训练参数的配置文件路径。"},
    },
    "device_count": {
        "en": {"label": "Device count", "info": "Number of devices available."},
        "zh": {"label": "设备数量", "info": "当前可用的运算设备数。"},
    },
    "output_box": {"en": {"label": "Info Box", "value": "Ready."}, "zh": {"label": "信息栏", "value": "准备就绪。"}},
    "loss_viewer": {"en": {"label": "Loss"}, "zh": {"label": "损失"}},
    "predict": {"en": {"label": "Save predictions"}, "zh": {"label": "保存预测结果"}},
    "infer_backend": {"en": {"label": "Inference engine"}, "zh": {"label": "推理引擎"}},
    "infer_dtype": {"en": {"label": "Inference data type"}, "zh": {"label": "推理数据类型"}},
    "load_btn": {"en": {"value": "Load model"}, "zh": {"value": "加载模型"}},
    "unload_btn": {"en": {"value": "Unload model"}, "zh": {"value": "卸载模型"}},
    "info_box": {
        "en": {"label": "Info", "value": "Model unloaded, please load a model first."},
        "zh": {"label": "信息栏", "value": "模型未加载，请先加载模型。"},
    },
    "role": {"en": {"label": "Role"}, "zh": {"label": "角色"}},
    "system": {"en": {"placeholder": "System prompt (optional)"}, "zh": {"placeholder": "系统提示词（非必填）"}},
    "tools": {"en": {"placeholder": "Tools (optional)"}, "zh": {"placeholder": "工具列表（非必填）"}},
    "image": {"en": {"label": "📷 Image"}, "zh": {"label": "📷 图像"}},
    "video": {"en": {"label": "📹 Video (optional)"}, "zh": {"label": "📹 视频"}},
    "query": {"en": {"placeholder": "Input..."}, "zh": {"placeholder": "输入..."}},
    "chat_btn": {"en": {"value": "💬 Chat"}, "zh": {"value": "💬 对话"}},
    "max_length": {"en": {"label": "Maximum length"}, "zh": {"label": "最大长度"}},
    "max_new_tokens": {"en": {"label": "Maximum new tokens"}, "zh": {"label": "最大生成长度"}},
    "top_p": {"en": {"label": "📊 Top-p"}, "zh": {"label": "📊 Top-p 采样值"}},
    "temperature": {"en": {"label": "🌡️ Temperature"}, "zh": {"label": "🌡️ 温度系数"}},
    "clear_btn": {"en": {"value": "Clear history"}, "zh": {"value": "清空历史"}},
    "export_size": {
        "en": {"label": "Max shard size (GB)", "info": "The maximum size for a model file."},
        "zh": {"label": "最大分块大小（GB）", "info": "单个模型文件的最大大小。"},
    },
    "export_quantization_bit": {
        "en": {"label": "Export quantization bit.", "info": "Quantizing the exported model."},
        "zh": {"label": "导出量化等级", "info": "量化导出模型。"},
    },
    "export_quantization_dataset": {
        "en": {"label": "Export quantization dataset", "info": "The calibration dataset used for quantization."},
        "zh": {"label": "导出量化数据集", "info": "量化过程中使用的校准数据集。"},
    },
    "export_device": {
        "en": {"label": "Export device", "info": "Which device should be used to export model."},
        "zh": {"label": "导出设备", "info": "导出模型使用的设备类型。"},
    },
    "export_legacy_format": {
        "en": {"label": "Export legacy format", "info": "Do not use safetensors to save the model."},
        "zh": {"label": "导出旧格式", "info": "不使用 safetensors 格式保存模型。"},
    },
    "export_dir": {
        "en": {"label": "Export dir", "info": "Directory to save exported model."},
        "zh": {"label": "导出目录", "info": "保存导出模型的文件夹路径。"},
    },
    "export_hub_model_id": {
        "en": {"label": "HF Hub ID (optional)", "info": "Repo ID for uploading model to Hugging Face hub."},
        "zh": {"label": "HF Hub ID（非必填）", "info": "用于将模型上传至 Hugging Face Hub 的仓库 ID。"},
    },
    "export_btn": {"en": {"value": "Export"}, "zh": {"value": "开始导出"}},
    "question_box": {
        "en": {"label": "💭 Question", "info": "Enter your question here..."},
        "zh": {"label": "💭 问题", "info": "在这里输入你的问题..."},
    },
    "seed_box": {"en": {"label": "🎲 Seed"}, "zh": {"label": "🎲 Seed"}},
    "question_type": {"en": {"label": "🌟 Question type"}, "zh": {"label": "🌟 问题类型"}},
    "state_checkbox_group": {
        "en": {
            "label": "Chat State",
            "info": "check whether ready to chat or not",
            "choices": ["Question", "Image", "Video", "Model"],
        },
        "zh": {"label": "对话状态", "info": "检查是否可以对话", "choices": ["问题", "图像", "视频", "模型"]},
    },
    "ckpt_box": {"en": {"label": "Checkpoint Item"}, "zh": {"label": "检查点"}},
}

ALERTS = {
    "err_conflict": {"en": "A process is in running, please abort it first.", "zh": "任务已存在，请先中断训练。"},
    "err_exists": {"en": "You have loaded a model, please unload it first.", "zh": "模型已存在，请先卸载模型。"},
    "err_no_model": {"en": "Please select a model.", "zh": "请选择模型。"},
    "err_no_cfg_path": {"en": "Please enter config path.", "zh": "请输入配置文件的路径"},
    "err_no_path": {"en": "Model not found.", "zh": "模型未找到。"},
    "err_no_dataset": {"en": "Please choose a dataset.", "zh": "请选择数据集。"},
    "err_no_adapter": {"en": "Please select an adapter.", "zh": "请选择适配器。"},
    "err_no_output_dir": {"en": "Please provide output dir.", "zh": "请填写输出目录。"},
    "err_failed": {"en": "Failed.", "zh": "训练出错。"},
    "err_json_schema": {"en": "Invalid JSON schema.", "zh": "Json 格式错误。"},
    "err_config_not_found": {"en": "Config file is not found.", "zh": "未找到配置文件。"},
    "warn_no_cuda": {"en": "CUDA environment was not detected.", "zh": "未检测到 CUDA 环境。"},
    "warn_output_dir_exists": {
        "en": "Output dir already exists, will resume training from here.",
        "zh": "输出目录已存在，将从该断点恢复训练。",
    },
    "info_aborting_error": {"en": "Aborted error during non-trainning phase", "zh": "非训练过程中，中断失败"},
    "info_aborting": {"en": "Aborted, wait for terminating...", "zh": "训练中断，正在等待进程结束……"},
    "info_aborted": {"en": "Aborted", "zh": "训练中断"},
    "info_finished": {"en": "Finished.", "zh": "训练完毕。"},
    "info_config_saved": {"en": "Arguments have been saved at: ", "zh": "训练参数已保存至："},
    "info_config_loaded": {"en": "Arguments have been restored.", "zh": "训练参数已载入。"},
    "info_loading": {"en": "Loading model...", "zh": "加载中……"},
    "info_training": {"en": "Training...", "zh": "训练中……"},
    "info_unloading": {"en": "Unloading model...", "zh": "卸载中……"},
    "info_loaded": {"en": "Model loaded!", "zh": "模型已加载！"},
    "info_unloaded": {"en": "Model unloaded.", "zh": "模型已卸载。"},
    "info_unload_error": {"en": "Model has not been loaded yet", "zh": "模型未加载，卸载失败"},
    "info_exporting": {"en": "Exporting model...", "zh": "正在导出模型……"},
    "info_exported": {"en": "Model exported.", "zh": "模型导出完成。"},
    "info_query": {"en": "Please enter your question and then try again.", "zh": "请输入你的问题后重新尝试。"},
    "info_upload_file": {"en": "Please upload your image or video before chatting", "zh": "请上传图片或视频后重新尝试"},
    "info_upload_model": {"en": "Please select and load your Model", "zh": "请选择并加载您的模型"},
    "info_generating": {"en": "Generating...", "zh": "生成中……"},
    "info_generated": {"en": "Generated successfully", "zh": "生成成功"},
}
