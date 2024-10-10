

import os
generated_cu = []
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".c") or file.endswith(".cu"):
            generated_cu.append(os.path.join(root, file))


import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]


gencode_flags = get_gencode_flags()



setup(
    name="triton_adaptive_layer_norm_fp16_2048_0_1_package",
    ext_modules=CUDAExtension(
        sources = generated_cu,
        extra_compile_args={
            "cc": ["-lcuda"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        },
        extra_link_args = ["-lcuda"]
    ),
)
