# Stable Diffusion 3 (SD3)

Stable Diffusion 3（SD3）是一种多模态扩散Transformer（MMDiT）文本生成图像模型，具有显著提升的图像质量、排版能力、复杂提示理解能力和资源效率。该模型能够根据文本提示生成高质量的图像。



与之前的版本相比，SD3 进行了以下两项重大改进：

1. **训练目标优化**：采用整流（rectified flow）替代原先DDPM中的训练目标，提升了训练效果。
2. **去噪模型升级**：将去噪模型从U-Net更换为更适合处理多模态信息的MM-DiT，增强了模型的多模态处理能力。

此外，SD3 在模型结构和训练目标上还进行了多项细微优化，包括调整训练噪声采样分布，并采用了三个固定的预训练文本编码器（OpenCLIP-ViT/G、CLIP-ViT/L 和 T5-xxl）。

<p align="center">
    <img src="https://private-user-images.githubusercontent.com/20476674/377710726-b5c8fea0-42e8-43d4-86ef-6488206f5855.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjkyMjI3NDUsIm5iZiI6MTcyOTIyMjQ0NSwicGF0aCI6Ii8yMDQ3NjY3NC8zNzc3MTA3MjYtYjVjOGZlYTAtNDJlOC00M2Q0LTg2ZWYtNjQ4ODIwNmY1ODU1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDE4VDAzMzQwNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNkMTFlYjE0N2I3MzFmNDA2MWQ2MDQ3YTVlZmQ0ODdlMTkxMTlkYzhmODI3NDAyNTFjNzZkOTQwNjAxZTJhNzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.vOqesJpEA2v0wVqMzxTBH69fZ46ij-Ckf14Ouy54ID0" alt="SD3模型结构" width="650">
</p>
<p align="center"><em>SD3模型结构图</em></p>

## 快速体验

想要快速体验 SD3 模型，可以参考以下推理示例：

- **文本生成图像**：[SD3 文本生成图像推理示例](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion_3.py)
- **图像生成图像**：[SD3 图像生成图像推理示例](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-stable_diffusion_3.py)

## 个性化微调

PPDiffusers 提供了 SD3 的个性化微调训练示例，仅需少量主题图像即可定制专属的 SD3 模型。支持以下微调方式：

- **DreamBooth LoRA 微调**
- **DreamBooth 全参数微调**

同时，支持在 NPU 硬件上进行训练。具体示例请参考：[DreamBooth 训练示例：Stable Diffusion 3 (SD3)](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/dreambooth/README_sd3.md)。

## 高性能推理

在推理方面，SD3 提供了高性能的推理实现，支持多卡并行推理。与竞品 TensorRT 相比，性能提升了 25.9%。具体内容请参考：[Stable Diffusion 3 高性能推理](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/deploy/sd3)。

---

欢迎使用 Stable Diffusion 3，期待您的反馈与贡献！