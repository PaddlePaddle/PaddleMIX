





- infer_static.py是静态图的推理代码
- 你需要导出两个模型，才可以使用静态图推理

- 为了静态图推理，你需要安装
    - develop版本的Paddle，cuda 需要至少11.2
    - PaddleNLP develop版本
- 指定了 --first_model_path和 --second_model_path之后你就可以推理了
