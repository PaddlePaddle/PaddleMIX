# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 是 Qwen 团队推出的一个专注于视觉与语言（Vision-Language, VL）任务的多模态大模型。它旨在通过结合图像和文本信息，提供强大的跨模态理解能力，可以处理涉及图像描述、视觉问答（VQA）、图文检索等多种任务。Qwen2-VL通过引入创新性的技术如 Naive Dynamic Resolution 和 M-RoPE，以及深入探讨大型多模态模型的潜力，显著地提高了多模态内容的视觉理解能力。

## 2 环境准备

- **python >= 3.10**
- **paddlepaddle-gpu 要求是develop版本**
```bash
# 安装示例
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/
```

- **paddlenlp 需要特定版本**

在PaddleMIX/代码目录下执行以下命令安装特定版本的paddlenlp：
```bash
# 安装示例
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
python setup.py install

# 此处提供两种paddlenlp_ops安装方法，建议使用预编译的paddlenlp_ops进行安装
# 手动编译安装paddlenlp_ops
cd csrc
python setup_cuda.py install

# 安装pre-build paddlenlp_ops
pip install https://paddlenlp.bj.bcebos.com/ops/cu118/paddlenlp_ops-3.0.0b4.post20250331-py3-none-any.whl
```

3） paddlenlp_ops预编译包安装表格，根据paddlenlp、CUDA版本选择配套paddlenlp_ops 

<table class="docutils">
    <thead>
        <tr>
            <th width="80">CUDA</th>
            <th width="200">paddlenlp_3.0.0b4</th>
            <th width="200">paddlenlp_develop</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">12.4</td>
            <td>
                <details>
                    <summary>Install</summary>
                    <pre><code>pip install https://paddlenlp.bj.bcebos.com/ops/cu124/paddlenlp_ops-3.0.0b4-py3-none-any.whl</code></pre>
                </details>
            </td>
            <td></td>
        </tr>
        <tr>
            <td align="center">11.8</td>
            <td>
                <details>
                    <summary>Install</summary>
                    <pre><code>pip install https://paddlenlp.bj.bcebos.com/ops/cu118/paddlenlp_ops-3.0.0b4-py3-none-any.whl</code></pre>
                </details>
            </td>
            <td>
                <details>
                    <summary>Install</summary>
                    <pre><code>pip install https://paddlenlp.bj.bcebos.com/ops/cu118/paddlenlp_ops-3.0.0b4.post20250331-py3-none-any.whl</code></pre>
                </details>
            </td>
        </tr>
    </tbody>
</table>

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 高性能推理

在Qwen2-VL的高性能推理优化中，**视觉模型部分继续使用PaddleMIX中的模型组网；但是语言模型部分调用PaddleNLP中高性能的Qwen2语言模型**，以得到高性能的Qwen2-VL推理版本。

### 3.1. 文本&单张图像输入高性能推理
```bash
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1
python deploy/qwen2_vl/single_image_infer.py\
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --append_attn True \
    --inference_model True \
    --llm_mode static \
    --dtype bfloat16 \
    --output_via_mq False \
    --benchmark True
```
### 3.2. 文本&视频输入高性能推理
```bash
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1
python deploy/qwen2_vl/video_infer.py\
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --question "Describe this video." \
    --video_file paddlemix/demo_images/red-panda.mp4 \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --append_attn True \
    --inference_model True \
    --llm_mode static \
    --dtype bfloat16 \
    --output_via_mq False \
    --benchmark True
```



## 4 一键推理 & 推理说明
```bash
cd PaddleMIX
sh deploy/qwen2_vl/scripts/qwen2_vl.sh
```
#### 参数设定：默认情况下，使用model自带的generation_config.json中的参数。
|     parameter      |      Value     |
| ------------------ | -------------- |
|       Top-K        |       1        |
|       Top-P        |     0.001      |
|    temperature     |      0.1       |
| repetition_penalty |      1.05      |

#### 单一测试demo执行时，指定max_length=min_length=128，固定输出长度。
|     parameter      |      Value     |
| ------------------ | -------------- |
|     min_length     |       128      |
|     min_length     |       128      |


#### 下方表格中所示性能对应的输入输出大小。
|     parameter            |      Value     |
| -------------------------| -------------- |
|  image_input_tokens_len  |  997 tokens    |
|  video_input_tokens_len  | 2725 tokens    |
|  output_tokens_len       |  128 tokens    |

- 在 NVIDIA A800-80GB 上测试的单图端到端速度性能如下：

| model                  | Paddle Inference wint8 | Paddle Inference|    PyTorch   |
| ---------------------- | ---------------------- | --------------- | ------------ |
| Qwen2-VL-2B-Instruct   |         0.636 s        |      0.777 s    |    2.086 s   |
| Qwen2-VL-7B-Instruct   |         1.121 s        |      1.677 s    |    3.132 s   |


- 在 NVIDIA A800-80GB 上测试的单视频端到端速度性能如下：

| model                  | Paddle Inference|    PyTorch   |
| ---------------------- | --------------- | ------------ |
| Qwen2-VL-2B-Instruct   |      1.306 s    |     3.143 s  |
| Qwen2-VL-7B-Instruct   |      2.337 s    |     2.715 s  |
