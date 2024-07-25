# PPDiffusers推理部署

PPDiffusers基于Paddle Inference，提供了以下重点扩散模型的部署方案:
- ControlNet
- Ip-Adapter-SD15
- Ip-Adapter-SDXL
- SD15
- SDXL


# V100性能数据
|模型|Paddle Deploy TensorRT / ips|Paddle Dynamic / ips|Torch Dynamic / ips|
|-|-|-|-|
|SD15 text2img|11.87|6.68|6.32|
|SD15 img2img|14.47|8.09|7.63|
|SD15 inpaint|14.30|6.42|6.06|


# A100性能数据
|模型|Paddle Deploy TensorRT|Paddle Dynamic|Torch Dynamic|
|-|-|-|-|
|SD15 text2img|26.37|10.49||
|SD15 img2img|30.81|12.70||
|SD15 inpaint|30.55|9.67||

<!-- |SDXL text2img||||
|SDXL img2img||||
|SDXL inpaint|||| -->

<!-- |-|-|-|-|
|ControlNet text2img|3.360597|||
|ControlNet img2img|3.360597|||
|ControlNet inpaint|3.360597||| -->