# PP-DocBee2高性能推理教程

## 1. 模型介绍

[PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2) 是PaddleMIX团队自研的一款专注于文档理解的多模态大模型，在PP-DocBee的基础上，我们进一步优化了基础模型，并引入了新的数据优化方案，提高了数据质量，使用自研[数据合成策略](https://arxiv.org/abs/2503.04065)生成的少量的47万数据便使得PP-DocBee2在中文文档理解任务上表现更佳。在内部业务中文场景类的指标上，PP-DocBee2相较于PP-DocBee提升了约11.4%，同时也高于目前的同规模热门开源和闭源模型。

| Model              | 模型大小 | Huggingface 仓库地址 |
|--------------------|----------|--------------------|
| PPDocBee2-3B | 3B | [PPDocBee2-3B](https://huggingface.co/PaddleMIX/PPDocBee2-3B) |

注意：使用`xxx.from_pretrained("PaddleMIX/PPDocBee2-3B")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备
1）
[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求develop版本**
```bash
# Develop 版本安装示例
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

```

2） [安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
```bash
# 安装paddlemix、ppdiffusers、项目依赖、PaddleNLP
sh build_env.sh --nlp_dev

# 此处提供两种paddlenlp_ops安装方法，建议使用预编译的paddlenlp_ops进行安装

# 手动编译安装paddlenlp_ops
cd PaddleNLP/csrc
python setup_cuda.py install

# 安装pre-build paddlenlp_ops
wget https://paddlenlp.bj.bcebos.com/wheels/paddlenlp_ops-ci-py3-none-any.whl -O paddlenlp_ops-0.0.0-py3-none-any.whl
pip install paddlenlp_ops-0.0.0-py3-none-any.whl
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

### a. 单卡高性能推理
```bash
cd PaddleMIX
rm -rf ./tmp

export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=512
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1
python deploy/ppdocbee2/ppdocbee2_infer.py \
    --model_name_or_path PaddleMIX/PPDocBee2-3B \
    --media_type "image" \
    --image_file "paddlemix/demo_images/medal_table.png" \
    --question "识别这份表格的内容, 以markdown格式输出" \
    --min_length 0 \
    --max_length 2048 \
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


### b. 多卡推理
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus "0,1" deploy/ppdocbee2/ppdocbee2_infer.py \
    --model_name_or_path PaddleMIX/PPDocBee2-3B \
    --media_type "image" \
    --question "识别这份表格的内容, 以markdown格式输出" \
    --image_file paddlemix/demo_images/medal_table.png \
    --min_length 0 \
    --max_length 2048 \
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
