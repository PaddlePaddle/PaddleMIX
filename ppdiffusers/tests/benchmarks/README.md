# PPdiffusers Benchmark 测试

## 测试原理
该目录下的脚本将自动遍历 `ppdiffusers/deploy` 下的各个模型文件夹，寻找其中的 scripts/benchmark_**backend**.sh 并执行。请提前为每个待测模型写好对应的 benchmark 测试脚本。

## 测试方式
首先，请打开该目录下的run_benchmarks_*.py脚本文件，指定项目文件夹路径，以及所使用的GPU。
接着运行以下代码进行批量benchmark性能测试：

### paddle_deploy_tensorrt 后端
速度比较慢，一般几个小时才能得到获得benchmark测试报告，推荐夜晚运行
```shell
stdbuf -oL -eL sh run_benchmarks_deploy_tensorrt.sh > trt.log &
```

### paddle_deploy 后端
速度稍慢，运行一小时以内即可获得benchmark测试报告
```shell
stdbuf -oL -eL sh run_benchmarks_deploy.sh > deploy.log &
```

### paddle 后端
速度较快
```shell
stdbuf -oL -eL sh run_benchmarks_paddle.sh > paddle.log &
```

### torch 后端
速度较快
```shell
stdbuf -oL -eL sh run_benchmarks_torch.sh > torch.log &
```

日志将记录在对应的 log 文件中。

**注：** 通过 `stdbuf` 运行脚本可以关闭日志缓冲，确保测试记录能够完整地重定向到指定的日志文件中。
