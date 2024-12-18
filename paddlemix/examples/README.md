
paddlemix `examples` 目录下提供模型的一站式体验，包括模型推理、模型静态图部署、模型预训练，调优等能力。帮助开发者快速了解 PaddleMIX 模型能力与使用方法，降低开发门槛。


## 支持模型

| Model                                           | Model Size                       | Template          |
|-------------------------------------------------| -------------------------------- | ----------------- |
| [YOLO-World](./YOLO-World/)                     | 640M/800M/1280M                  | yolo_world        |
| [audioldm2](./audioldm2/)                       | 346M/712M                        | audioldm2         |
| [blip2](./blip2/)                               | 7B                               | blip2             |
| [clip](./clip)                                  | 2539.57M/1366.68M/986.71M/986.11M/427.62M/149.62M/151.28M | clip              |
| [coca](./coca/)                                 | 253.56M/638.45M/253.56M/638.45M  | coca              |
| [CogVLM && CogAgent](./cogvlm/)                 | 17B                              | cogvlm_cogagent   |
| [eva02](./eva02/)                               | 6M/22M/86M/304M                  | eva02             |
| [evaclip](./evaclip/)                           | 1.1B/1.3B/149M/428M/4.7B/5.0B    | evaclip           |
| [groundingdino](./groundingdino/)               | 172M/341M                        | groundingdino     |
| [imagebind](./imagebind/)                       | 1.2B                             | imagebind         |
| [InternLM-XComposer2](./internlm_xcomposer2/)   | 7B                               | internlm_xcomposer2 |
| [Internvl2](./internvl2/)                       | 1B/2B/8B/26B/40B                 | internvl2         |
| [janus](./janus/)                               | 1.3B                             | janus             |
| [llava](./llava/)                               | 7B/13B                           | llava             |
| [llava_critic](./llava_critic/)                 | 7B                               | llava_critic      |
| [llava_denseconnector](./llava_denseconnector/) | 7B                               | llava_denseconnector |
| [llava_next](./llava_next_interleave/)          | 0.5B/7B                          | llava_next_interleave |
| [llava_onevision](./llava_onevision/)           | 0.5B/2B/7B                       | llava_onevision   |
| [minicpm-v-2_6](./minicpm_v_2_6/)               | 8B                               | minicpm_v_2_6     |
| [minigpt4](./minigpt4/)                         | 7B/13B                           | minigpt4          |
| [minimonkey](./minimonkey/)                     | 2B                               | minimonkey        |
| [qwen2_vl](./qwen2_vl/)                         | 2B/7B/72B                        | qwen2_vl          |
| [qwen_vl](./qwen_vl/)                           | 7B                               | qwen_vl           |
| [sam](./sam/)                                   | 86M/307M/632M                    | sam               |
| [visualglm](./visualglm/)                       | 6B                               | visualglm         |


## 模型能力矩阵

| Model                                           | Inference | Pretrain | SFT | LoRA | Deploy | NPU Training |
|-------------------------------------------------| --------- | -------- | --- | ---- | ------ | ------------ |
| [YOLO-World](./YOLO-World/)                     | ✅        | ❌       | ❌  | ❌   | ❌     | ❌           |
| [audioldm2](./audioldm2/)                       | ✅        | ❌       | ❌  | ❌   | ❌     | ❌           |
| [blip2](./blip2/)                               | ✅        | ✅      | ✅  | ✅   | ❌     | ❌           |
| [clip](./clip)                                  | ✅        | ✅      | ❌  | ❌   | ❌     | ❌           |
| [coca](./coca/)                                 | ✅        | ✅      | ❌  | ❌   | ❌     | ❌           |
| [CogVLM && CogAgent](./cogvlm/)                 | ✅        | ❌       | ❌  | ❌   | ❌     | ❌           |
| [eva02](./eva02/)                               | ✅        | ✅      | ✅  | ❌   | ❌     | ❌           |
| [evaclip](./evaclip/)                           | ✅        | ✅      | ❌  | ❌   | ❌     | ❌           |
| [groundingdino](./groundingdino/)               | ✅        | ❌       | 🚧  | ❌   | ✅     | ❌           |
| [imagebind](./imagebind/)                       | ✅        | ❌       | ❌  | ❌   | ❌     | ❌           |
| [InternLM-XComposer2](./internlm_xcomposer2/)   | ✅ | ❌ | ✅  | ❌   | ❌     | ❌           |
| [Internvl2](./internvl2/)                       | ✅        | ❌       | ✅  | ❌   | ❌     | ✅           |
| [janus](./janus/)                               | ✅        | ❌       | ❌  | ❌   | ❌     | ❌            |
| [llava](./llava/)                               | ✅        | ✅      | ✅  | ✅   | 🚧    | ✅           |
| [llava_critic](./llava_critic/)                 | ✅        | ❌       | ❌  | ❌   | ❌     | ❌           |
| [llava_denseconnector](./llava_denseconnector/) | ✅ | ❌ | ❌  | ❌   | ❌     | ❌           |
| [llava_next](./llava_next_interleave/)          | ✅ | ❌ | ❌  | ❌   | ❌     | ❌           |
| [llava_onevision](./llava_onevision/)           | ✅       | ❌       | ❌  | ❌   | ❌     | ❌           |
| [minicpm-v-2_6](./minicpm_v_2_6/)               | ✅        | ❌       | ❌  | ❌   | ❌     | ❌           |
| [minigpt4](./minigpt4/)                         | ✅        | ✅      | ✅  | ❌   | ✅     | ❌           |
| [minimonkey](./minimonkey/)                     | ✅        | ❌       | ✅  | ❌   | ❌     | ❌           |
| [qwen2_vl](./qwen2_vl/)                         | ✅        | ❌       | ✅  | 🚧  | ❌     | ❌           |
| [qwen_vl](./qwen_vl/)                           | ✅        | ❌       | ✅  | ✅   | ✅     | ❌           |
| [sam](./sam/)                                   | ✅        | ❌       | ❌  | ❌   | ✅     | ❌           |
| [visualglm](./visualglm/)                       | ✅        | ❌       | ✅  | ✅   | ❌     | ❌           |


>* ✅: Supported
>* 🚧: In Progress
>* ❌: Not Supported
