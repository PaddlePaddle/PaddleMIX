# PP-VCtrl 高性能推理

- Paddle Inference提供 PP-VCtrl 系列模型的高性能推理实现，推理性能提升10%+
环境准备：

```shell
# 安装 triton并适配paddle
python -m pip install triton
python -m pip install git+https: //github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"

# 安装develop版本的paddle，请根据自己的cuda版本选择对应的paddle版本，这里选择12.3的cuda版本
python -m pip install --pre paddlepaddle-gpu -i https: //www.paddlepaddle.org.cn/packages/nightly/cu123/

# 安装paddlemix库,使用集成在paddlemix库中的自定义算子。
python -m pip install paddlemix

```

## 推理优化内容：
目前PP-VCtrl通过使用部分高性能的融合算子来提升推理性能，例如`ln_partial_rotary_emb`和`partial_rotary_emb`等高性能融合算子。
其中，`ln_partial_rotary_emb`将Q，K的Norm以及ROPE算子融合；

## 高性能推理指令：
```shell
cd ppdiffusers/examples/PP-VCtrl/deploy
bash scripts/infer_cogvideox_i2v_pose_vctrl.sh
```
注：--inference_optimize 1 用于开启推理优化，--benchmark 1 用于开启性能测试。


## 实测性能
- 在 NVIDIA A800-80GB 上测试得端到端速度性能如下：

|                     model                  | Paddle Inference | Paddle 动态图 |
| ------------------------------------------ | ---------------  | ------------ |
| cogvideox-5b-i2v-vctrl & vctrl_pose_5b_i2v |     122.184 s    |   136.375 s  |
