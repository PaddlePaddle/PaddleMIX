# PPdiffusers Benchmark 测试

## 测试原理
该目录下的脚本将自动遍历 `ppdiffusers/deploy` 下的各个模型文件夹，寻找其中的 `scripts/benchmark_backend.sh` 并执行。请提前为每个模型注册并定义好相应的 benchmark 测试脚本。

## 测试方式
首先，请打开脚本文件制定项目文件夹路径，以及所使用的GPU。

### paddle_deploy_tensorrt 后端
```shell
stdbuf -oL -eL sh run_benchmarks_deploy_tensorrt.sh > trt.log &
```

### paddle_deploy 后端
```shell
stdbuf -oL -eL sh run_benchmarks_deploy.sh > deploy.log &
```

### paddle 后端
```shell
stdbuf -oL -eL sh run_benchmarks_paddle.sh > paddle.log &
```

### torch 后端
```shell
stdbuf -oL -eL sh run_benchmarks_torch.sh > torch.log &
```

日志将记录在对应的 log 文件中。

**注：** 通过 `stdbuf` 运行脚本可以关闭日志缓冲，确保测试记录能够完整地重定向到指定的日志文件中。
