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

# 指定 libCutlassGemmEpilogue.so 的路径
# 详情请参考 https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/fusion/cutlass/gemm_epilogue/README.md
export LD_LIBRARY_PATH=/your_dir/Paddle/paddle/phi/kernels/fusion/cutlass/gemm_epilogue/build:$LD_LIBRARY_PATH
```

高性能推理指令：
```shell
# 执行FP16推理
python  text_to_image_generation-stable_diffusion_3.py  --dtype float16 --height 512 --width 512 \
--num-inference-steps 50 --inference_optimize 1  \
--benchmark 1
```

- 在 NVIDIA A100-SXM4-40GB 上测试的性能如下：

| Paddle Inference|    PyTorch   | Paddle 动态图 |
| --------------- | ------------ | ------------ |
|       1.2 s     |     1.78 s   |    4.202 s   |


## Paddle Stable Diffusion 3 模型多卡推理： 
### batch parallel 实现原理  
- 由于MMDIT输入的prompt的batch为2，所以我们考虑在多卡并行的方案中，将batch为2的输入拆分到两张卡上进行计算，达到并行计算的效果，降低单卡所承载的浮点计算量，提升推理性能。
### 开启多卡推理方法 
- Paddle Inference 提供了SD3模型的多卡推理功能，用户可以通过设置 `--inference_optimize_bp 1` 来开启这一功能，  
使用 `python -m paddle.distributed.launch --gpus 0,1` 指定使用哪些卡进行推理。
高性能多卡推理指令：
```shell
# 执行多卡推理指令
python -m paddle.distributed.launch --gpus 0,1 text_to_image_generation-stable_diffusion_3.py \
--dtype float16 \
--height 512 --width 512 \
--num-inference-steps 50 \
--inference_optimize 1 \
--inference_optimize_bp 1 \
--benchmark 1
```
## 在 NVIDIA A800-SXM4-80GB 上测试的性能如下：

| Paddle batch parallel | Paddle Single Card |  PyTorch  | Paddle 动态图 |
| --------------------- | ------------------ | --------- | ------------ |
|          0.86 s       |        1.2 s       |   1.78 s  |    4.202 s   |
