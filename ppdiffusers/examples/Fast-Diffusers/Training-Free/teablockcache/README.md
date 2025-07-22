# TeaBlockCache

## 快速简介

> **TeaBlockCache** 在 TeaCache 的时间维缓存框架上进一步细化到了 Block 级：对每个 Block 保存上一次经 Layer Norm 调制后的输入快照，仅取前 1/8 通道计算 L1 相对变化，并将变化经多项式加权累积为误差积分；若积分未超阈值则直接复用缓存结果、跳过该 Block 计算，否则刷新缓存并重新前向。同时，系统维护一份全局泰勒缓存：每次真实前向都把当前隐状态写入 0 阶项，并用差分近似更新 1–2 阶导数；当首块连续多步波动极小且超过 first_enhance 阈值时，触发整网泰勒外推，用缓存导数预测下一步隐状态，若预测数值稳定则整步前向全部省略。这样，框架能在细粒度的 Block 级跳算与粗粒度的整网外推之间自适应切换，在保持生成质量的同时将推理时间进一步压缩至 2× 以上。


## 使用方法
```
python text_to_image_generation_teablockcache_taylor_flux.py
```

或通过hook方式调用：
```
python text_to_image_generation_teablockcache_taylor_flux_hook.py
```

## 参数详解

| 字段                         | 类型   | 作用 / 触发条件                                                   | 常用取值 |
|------------------------------|--------|-------------------------------------------------------------------|----------|
| `step_start`                 | int    | Block/Taylor 跳算 **开始** 的时间步（此前全量计算）                 | `50–100` |
| `step_end`                   | int    | Block/Taylor 跳算 **结束** 的时间步（之后恢复全量计算）             | `900–950`|
| `block_cache_start`          | int    | Block-级缓存启用起始步                                            | `1`      |
| `single_block_cache_start`   | int    | 单 Block 缓存启用起始步                                           | `1`      |
| `block_rel_l1_thresh`        | float  | 判断整块可否复用的 **L1 相对阈值**                                 | `1–3`    |
| `single_block_rel_l1_thresh` | float  | 判断单块可否复用的 **L1 相对阈值**                                 | `1–3`    |
| `rel_l1_thresh`              | float  | TeaCache **全局** L1 阈值                                         | `1–3`    |
| `max_order`                  | int    | 泰勒缓存 **最高阶**（0=值，1=一阶导 …）                            | `1` 或 `2`|
| `first_enhance`              | int    | 首块连续极小波动达到此次数后触发整网泰勒外推                      | `1–3`    |
