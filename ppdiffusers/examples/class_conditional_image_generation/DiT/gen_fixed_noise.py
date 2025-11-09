import paddle
import numpy as np

paddle.seed(1234)  # 固定随机种子
noise = paddle.randn([5000, 4, 32, 32], dtype="float32")  # 举例：形状按你模型需要调整
# np.save("/path/to/dit_fixed_noise_B5000.npy", noise.numpy())  # 保存为 .npy
# # 或保存为 .pdparams
paddle.save({"fixed_noise": noise}, "/share/chenqian-local/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/dit_fixed_noise_B5000.pdparams")