# PPDiffusers推理部署

PPDiffusers基于Paddle Inference，提供了以下重点扩散模型的部署方案:
- ControlNet
- IP-Adapter-SD15
- IP-Adapter-SDXL
- SD15
- SDXL


# V100性能数据
|模型|Paddle Deploy TensorRT / ips|Paddle Dynamic / ips|Torch Dynamic / ips|
|-|-|-|-|
|IP-Adapter-SD15 text2img|18.30|15.27|18.18|
|IP-Adapter-SD15 img2img|18.11|15.34|17.87|
|IP-Adapter-SD15 inpaint|17.93|14.96|17.44|
|IP-Adapter-SDXL text2img|12.01|8.93|11.47|
|IP-Adapter-SDXL img2img|12.00|8.72|10.95|
|IP-Adapter-SDXL inpaint|11.67|8.51|10.79|
|SD15 text2img|19.68|15.98|18.27|
|SD15 img2img|19.68|15.77|17.90|
|SD15 inpaint|19.44|15.54|17.56|
|SDXL text2img|13.91|9.31|11.50|
|SDXL img2img|13.86|8.82|11.60|
|SDXL inpaint|13.45|8.79|11.28|

<!-- |SD15 text2img|11.87|6.68|6.32|
|SD15 img2img|14.47|8.09|7.63|
|SD15 inpaint|14.30|6.42|6.06| -->

> Note: 
> 测试环境或配置为Paddle 3.0，V100 32G单卡，FP16。
推理参数为Image Width = 512， Image Height = 512， Num Inference Steps = 50。

# A100性能数据
|模型|Paddle Deploy TensorRT|Paddle Dynamic|
|-|-|-|
|IP-Adapter-SD15 text2img|38.52|22.76||
|IP-Adapter-SD15 img2img|37.91|22.54||
|IP-Adapter-SD15 inpaint|37.80|22.28||
|IP-Adapter-SDXL text2img|22.88|13.30||
|IP-Adapter-SDXL img2img|22.79|13.00||
|IP-Adapter-SDXL inpaint|22.30|13.20||
|SD15 text2img|47.22|23.73||
|SD15 img2img|46.59|23.75||
|SD15 inpaint|46.05|23.45||
|SDXL text2img|31.98|13.94||
|SDXL img2img|31.80|13.64||
|SDXL inpaint|30.58|12.30||

<!-- |SD15 text2img|26.37|10.49||
|SD15 img2img|30.81|12.70||
|SD15 inpaint|30.55|9.67|| -->

> Note: 测试环境或配置为Paddle 3.0，A100 80G单卡，FP16。
推理参数为Image Width = 512， Image Height = 512， Num Inference Steps = 50。

<!-- |SDXL text2img||||
|SDXL img2img||||
|SDXL inpaint|||| -->

<!-- |-|-|-|-|
|ControlNet text2img|3.360597|||
|ControlNet img2img|3.360597|||
|ControlNet inpaint|3.360597||| -->