1. 启动方式
在PaddleMix文件夹下执行`dist_train.sh`脚本即可启动

2. 相关准备
2.1 数据准备
1) 准备数据列表:filelists,放置在`PaddleMix`根目录下。
2）准备数据软连接data文件夹
3) 启动时使用——-train_data指定数据列表文件

2.2 模型
`.training_24l.conf`中`MODEL_NAME`设置所需创建的模型，代码中通过`EVA.from_pretrained(MODEL_NAME)`命令创建，有两种创建方法，
    1）MODEL_NAME指定为模型名称，例`EVA/EVA02-CLIP-B-16`; 
    2)指定绝对路径，根据路径下`config.json`和`model_state.pdparams`创建。`config.json`中设置模型相关结构参数。(类似开源config文件，注意：部分参数名称不同，原因是采用了类参数名称)

3. 相关设置修改：
3.1 分布式相关设置，在`.training_24l.conf`中`Distributed`部分。
要求：
```
 TRAINERS_NUM * TRAINING_GPUS_PER_NODE == DP_DEGREE * MP_DEGREE * SHARDING_DEGREE
```
暂时未验证添加`PP_DEGREE`效果

3.2 MP时需要设置预训练模型参数`PRETRAINED_MODEL`，需要单独的分布式模型切分功能，会根据该设置单独加载一次模型

3.3 优化器、学习率等在中`Base Set`部分设置





