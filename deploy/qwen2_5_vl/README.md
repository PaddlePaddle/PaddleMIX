# Qwen2.5-VL

## 1. 模型介绍

[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) 是 Qwen 团队推出的一个专注于视觉与语言（Vision-Language, VL）任务的多模态大模型。它旨在通过结合图像和文本信息，提供强大的跨模态理解能力，可以处理涉及图像描述、视觉问答（VQA）、图文检索等多种任务。

| Model              |
|--------------------|
| Qwen/Qwen2.5-VL-3B-Instruct  |
| Qwen/Qwen2.5-VL-7B-Instruct  |
| Qwen/Qwen2.5-VL-72B-Instruct  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备
1）
[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求develop版本**
```bash
# Develop 版本安装示例
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/

```

2） [安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
```bash
# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖
python -m pip install -e . --user
python -m pip install -e ppdiffusers --user
python -m pip install -r requirements.txt --user

# 安装PaddleNLP
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

## 3 高性能推理

### a. fp16 高性能推理
```bash
cd PaddleMIX
rm -rf ./tmp

# 1. image infer
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --media_type "image" \
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

# 2. video infer
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --media_type "video" \
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


### b. wint8 高性能推理
```bash
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --media_type "image" \
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
    --quant_type "weight_only_int8" \
    --output_via_mq False \
    --benchmark True
```

### c. TP并行，多卡高性能推理
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus "0,1" deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --media_type "image" \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --inference_model True \
    --mode dynamic \
    --append_attn 1 \
    --dtype bfloat16 \
    --output_via_mq False \
    --benchmark True
```


## 4 一键推理 & 推理说明
```bash
cd PaddleMIX
sh deploy/qwen2_5_vl/scripts/qwen2_5_vl.sh
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

## 在 NVIDIA A800-SXM4-80GB 上测试的性能如下：

#### 下方表格中所示性能对应的输入输出大小。
|     parameter            |      Value     |
| -------------------------| -------------- |
|  image_input_tokens_len  |  997 tokens    |
|  video_input_tokens_len  | 2725 tokens    |
|  output_tokens_len       |  128 tokens    |


#### 单图测试性能比较
|             model           | Paddle Inference wint8 | Paddle Inference|    PyTorch     | VLLM          |
| --------------------------- | ---------------------  | --------------- | -------------- |-------------- |
| Qwen/Qwen2.5-VL-3B-Instruct |          0.823 s       |     1.15 s      |      4.92 s    |    1.39s      |
| Qwen/Qwen2.5-VL-7B-Instruct |          1.144 s       |     1.16 s      |      3.89 s    |    1.70s      |
| Qwen/Qwen2.5-VL-72B-Instruct|             -          |     4.806 s     |        -       |      -        |


#### 单视频测试性能比较

|             model           | Paddle Inference wint8 | Paddle Inference|    PyTorch     | VLLM          |
| --------------------------- | ---------------------  | --------------- | -------------- |-------------- |
| Qwen/Qwen2.5-VL-3B-Instruct |          1.26 s        |     1.62 s      |      6.07 s    |    3.41s      |
| Qwen/Qwen2.5-VL-7B-Instruct |          1.59 s        |     2.11 s      |      4.59 s    |    2.82s      |