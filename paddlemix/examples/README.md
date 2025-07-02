
paddlemix `examples` 目录下提供模型的一站式体验，包括模型推理、模型静态图部署、模型预训练，调优等能力。帮助开发者快速了解 PaddleMIX 模型能力与使用方法，降低开发门槛。

## 支持模型
<table align="center">
  <tbody>
    <tr align="center" valign="center">
        <td>支持能力 </td>
        <td>Model</td>
        <td>Model Size</td>
        <td>Template</td>
    </tr>
    <tr align="center" valign="center">
        <td rowspan="14"> 一站式训推模型 </td>
        <td> <a href="./ppdocbee"> ppdocbee </a></td>
        <td> 2B/7B</td>
        <td> ppdocbee </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./qwen2_vl/"> qwen2_vl </a></td>
        <td> 2B/7B/72B </td>
        <td> qwen2_vl </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./internvl2/"> Internvl2 </a></td>
        <td> 1B/2B/8B/26B/40B </td>
        <td> internvl2 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./qwen_vl/"> qwen_vl </a></td>
        <td> 7B </td>
        <td> qwen_vl </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./minimonkey/">minimonkey </a></td>
        <td> 2B </td>
        <td>minimonkey</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./minigpt4/">minigpt4 </a></td>
        <td> 7B/13B </td>
        <td>minigpt4</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./visualglm/">visualglm </a></td>
        <td> 6B </td>
        <td>visualglm</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./internlm_xcomposer2/">InternLM-XComposer2 </a></td>
        <td> 7B </td>
        <td>internlm_xcomposer2</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava/">llava </a></td>
        <td> 7B </td>
        <td>llava</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./eva02/">eva02 </a></td>
        <td> 6M/22M/86M/304M </td>
        <td>eva02</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./evaclip/">evaclip </a></td>
        <td> 1.1B/1.3B/149M/428M/4.7B/5.0B </td>
        <td>evaclip</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./coca/">coca </a></td>
        <td> 253.56M/638.45M/253.56M/638.45M </td>
        <td>coca</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./blip2/">blip2 </a></td>
        <td> 7B </td>
        <td>blip2</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./clip/">clip </a></td>
        <td> 2539.57M/1366.68M/986.71M/986.11M/427.62M/149.62M/151.28M </td>
        <td>clip</td>
    </tr>
    <tr align="center" valign="center">
        <td rowspan="34"> 快速上手体验模型 </td>
        <td> <a href="./deepseek_vl2/"> deepseek_vl2 </a></td>
        <td> 3B/16B/27B    </td>
        <td> deepseek_vl2 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./aria/">aria </a></td>
        <td>24.9B</td>
        <td>aira</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./ppdocbee"> ppdocbee </a></td>
        <td> 2B/7B</td>
        <td> ppdocbee </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./GOT_OCR_2_0/">GOT_OCR_2_0 </a></td>
        <td> 0.6B </td>
        <td> GOT_OCR_2_0 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./sam/">sam </a></td>
        <td> 86M/307M/632M </td>
        <td> sam </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./sam2/">sam2 </a></td>
        <td> 224M </td>
        <td> sam2 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./audioldm2/">audioldm2 </a></td>
        <td> 346M/712M </td>
        <td> audioldm2 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./diffsinger/">diffsinger </a></td>
        <td> 80M </td>
        <td> diffsinger </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./janus/">janus </a></td>
        <td> 1B/1.3B/7B </td>
        <td> janus </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./emu3/">emu3 </a></td>
        <td> 8B </td>
        <td> emu3 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./showo/">showo </a></td>
        <td> 1.3B </td>
        <td> showo </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./mPLUG_Owl3/">mPLUG_Owl3 </a></td>
        <td>7B </td>
        <td>mPLUG_Owl3</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./minicpm-v-2_6">minicpm_v_2_6 </a></td>
        <td>8B </td>
        <td>minicpm_v_2_6</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./imagebind/">imagebind </a></td>
        <td>1.2B </td>
        <td>imagebind</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./YOLO-World/">YOLO-World </a></td>
        <td> 640M/800M/1280M</td>
        <td> yolo_world </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./groundingdino/">groundingdino </a></td>
        <td>172M/341M    </td>
        <td>groundingdino</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./cogvlm/">CogVLM && CogAgent </a></td>
        <td>17B</td>
        <td>cogvlm_cogagent</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./qwen2_vl/"> qwen2_vl </a></td>
        <td> 2B/7B/72B </td>
        <td> qwen2_vl </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./internvl2/"> Internvl2 </a></td>
        <td> 1B/2B/8B/26B/40B </td>
        <td> internvl2 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./qwen_vl/"> qwen_vl </a></td>
        <td> 7B </td>
        <td> qwen_vl </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./minimonkey/">minimonkey </a></td>
        <td> 2B </td>
        <td>minimonkey</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./minigpt4/">minigpt4 </a></td>
        <td> 7B/13B </td>
        <td>minigpt4</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./visualglm/">visualglm </a></td>
        <td> 7B/13B </td>
        <td>visualglm</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./internlm_xcomposer2/">InternLM-XComposer2 </a></td>
        <td> 7B </td>
        <td>internlm_xcomposer2</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava/">llava </a></td>
        <td> 7B </td>
        <td>llava</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava_onevision/">llava_onevision </a></td>
        <td> 0.5B/2B/7B </td>
        <td>llava_onevision</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava_next_interleave/">llava_next </a></td>
        <td> 0.5B/7B </td>
        <td>llava_next_interleave</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava_denseconnector/">llava_denseconnector </a></td>
        <td> 7B </td>
        <td>llava_denseconnector </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava_critic/">llava_critic </a></td>
        <td> 7B </td>
        <td>llava_critic </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./eva02/">eva02 </a></td>
        <td> 6M/22M/86M/304M </td>
        <td>eva02</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./evaclip/">evaclip </a></td>
        <td> 1.1B/1.3B/149M/428M/4.7B/5.0B </td>
        <td>evaclip</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./coca/">coca </a></td>
        <td> 253.56M/638.45M/253.56M/638.45M </td>
        <td>coca</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./blip2/">blip2 </a></td>
        <td> 7B </td>
        <td>blip2</td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./clip/">clip </a></td>
        <td> 2539.57M/1366.68M/986.71M/986.11M/427.62M/149.62M/151.28M </td>
        <td>clip</td>
    </tr>
    <tr align="center" valign="center">
        <td rowspan="3"> NPU训推支持模型 </td>
        <td> <a href="./qwen2_vl/"> qwen2_vl </a></td>
        <td> 2B/7B/72B </td>
        <td> qwen2_vl </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./internvl2/"> Internvl2 </a></td>
        <td> 1B/2B/8B/26B/40B </td>
        <td> internvl2 </td>
    </tr>
    <tr align="center" valign="center">
        <td> <a href="./llava/">llava </a></td>
        <td> 7B </td>
        <td>llava</td>
    </tr>
</tbody>
</table>
