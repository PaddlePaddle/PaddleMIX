# Stable Diffusion 3 高性能推理

- Paddle Inference提供Stable Diffusion 3 模型高性能推理实现，推理性能提升70%+
环境准备：
```shell
# 安装 triton并适配paddle
python -m pip install triton
python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"

# 安装develop版本的paddle，请根据自己的cuda版本选择对应的paddle版本，这里选择12.3的cuda版本
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/

# 安装paddlemix库,使用集成在paddlemix库中的自定义算子。
python -m pip install paddlemix

# 指定 libCutlassGemmEpilogue.so 的路径
# 详情请参考 https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/fusion/cutlass/gemm_epilogue/README.md
export LD_LIBRARY_PATH=/your_dir/Paddle/paddle/phi/kernels/fusion/cutlass/gemm_epilogue/build:$LD_LIBRARY_PATH
- 请注意，该项用于在静态图推理时利用Cutlass融合算子提升推理性能，但是并不是必须项。
如果不使用Cutlass可以将`./text_to_image_generation-stable_diffusion_3.py`中的`exp_enable_use_cutlass`设为False。
-
```

高性能推理指令：
```shell
# 执行FP16推理
python  text_to_image_generation-stable_diffusion_3.py  --dtype float16 --height 512 --width 512 \
--num-inference-steps 50 --inference_optimize 1  \
--benchmark 1
```
注：--inference_optimize 1 用于开启推理优化，--benchmark 1 用于开启性能测试。


- 在 NVIDIA A100-SXM4-40GB 上测试的性能如下：

| Paddle Inference|    PyTorch   | Paddle 动态图 |
| --------------- | ------------ | ------------ |
|       1.2 s     |     1.78 s   |    4.202 s   |


## Paddle Stable Diffusion 3 模型多卡推理：
### Data Parallel 实现原理
- 在SD3中，对于输入是一个prompt时，使用CFG需要同时进行unconditional guide和text guide的生成，此时 MM-DiT-blocks 的输入batch_size=2；
所以我们考虑在多卡并行的方案中，将batch为2的输入拆分到两张卡上进行计算，这样单卡的计算量就减少为原来的一半，降低了单卡所承载的浮点计算量。
计算完成后，我们再把两张卡的计算结果聚合在一起，结果与单卡计算完全一致。

### Model parallel 实现原理
- 在SD3中,在Linear和Attnetion中有大量的GEMM（General Matrix Multiply），当生成高分辨率图像时，GEMM的计算量以及模型的预训练权重大小都呈线性递增。
因此，我们考虑在多卡并行方案中，将模型的这些GEMM拆分到两张卡上进行计算，这样单卡的计算量和权重大小就都减少为原来的一半，不仅降低了单卡所承载的浮点计算量，也降低了单卡的显存占用。

### 开启多卡推理方法
- Paddle Inference 提供了SD3模型的多卡推理功能，用户可以通过设置 `mp_size 2` 来开启Model Parallel，使用 `dp_size 2`来开启Data Parallel。
使用 `python -m paddle.distributed.launch --gpus “0,1,2,3”` 指定使用哪些卡进行推理，其中`--gpus “0,1,2,3”`即为启用的GPU卡号。
如果只需使用两卡推理，则只需指定两卡即可，如 `python -m paddle.distributed.launch --gpus “0,1”`。同时需要指定使用的并行方法及并行度，如 `mp_size 2` 或者 `dp_size 2`。

- 注意，这里的`mp_size`需要设定为不大于输入的batch_size个，且`mp_size`和`dp_size`的和不能超过机器总卡数。
- 高性能多卡推理指令：
```shell
# 执行多卡推理指令
python -m paddle.distributed.launch --gpus "0,1,2,3" text_to_image_generation-stable_diffusion_3.py \
--dtype float16 \
--height 1024 \
--width 1024 \
--num-inference-steps 20 \
--inference_optimize 1 \
--mp_size 2 \
--dp_size 2 \
--benchmark 1
```
注：--inference_optimize 1 用于开启推理优化，--benchmark 1 用于开启性能测试。

## 在 NVIDIA A800-SXM4-80GB 上测试的性能如下：

| Paddle mp_size=2 & dp_size=2 |  Paddle mp_size=2   | Paddle dp_size=2 | Paddle Single Card | Paddle 动态图 |
| ---------------------------- | ------------------- | ---------------- | ------------------ | ------------ |
|            0.99s             |        1.581 s      |      1.319 s     |       2.376 s      |     3.2 s    |
