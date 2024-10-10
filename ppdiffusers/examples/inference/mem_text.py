# export CUDA_VISIBLE_DEVICES=0
import paddle
paddle.device.set_device('gpu')




cuda_mem_after_used = paddle.device.cuda.max_memory_allocated(0) / (1024**3)
print(f"max_memory_allocated : {cuda_mem_after_used:.3f} GiB")

tensor = paddle.randn([512, 512, 512], "float")
tensor0 = paddle.randn([512, 512, 512], "float")
tensor1 = paddle.add(tensor,tensor0)
tensor1 = paddle.add(tensor,tensor0)



#del tensor
#paddle.device.cuda.empty_cache()

cuda_mem_after_used1 = paddle.device.cuda.memory_reserved(0) / (1024**3)
print(f"memory_reserved : {cuda_mem_after_used1:.3f} GiB")


exit(0)


cuda_mem_after_used = paddle.device.cuda.max_memory_allocated(0) / (1024**3)
print(f"max_memory_allocated : {cuda_mem_after_used:.3f} GiB")

import time
time.sleep(1000)

tensor1 = paddle.randn([512, 512, 512], "float")

cuda_mem_after_used1 = paddle.device.cuda.max_memory_reserved(0) / (1024**3)
print(f"max_memory_reserved : {cuda_mem_after_used1:.3f} GiB")

tensor2 = paddle.randn([512, 512, 512], "float")

cuda_mem_after_used6 = paddle.device.cuda.memory_allocated(0) / (1024**3)
print(f"memory_allocated : {cuda_mem_after_used6:.3f} GiB")

tensor3 = paddle.randn([512, 512, 512], "float")

cuda_mem_after_used3 = paddle.device.cuda.memory_reserved(0) / (1024**3)
print(f"memory_reserved : {cuda_mem_after_used3:.3f} GiB")



del tensor1
del tensor2
print("--------------------------")
paddle.device.cuda.empty_cache()
cuda_mem_after_used4 = paddle.device.cuda.max_memory_allocated(0) / (1024**3)
print(f"max_memory_allocated : {cuda_mem_after_used4:.3f} GiB")
cuda_mem_after_used5 = paddle.device.cuda.max_memory_reserved(0) / (1024**3)
print(f"max_memory_reserved : {cuda_mem_after_used5:.3f} GiB")
cuda_mem_after_used6 = paddle.device.cuda.memory_allocated(0) / (1024**3)
print(f"memory_allocated : {cuda_mem_after_used6:.3f} GiB")
cuda_mem_after_used7 = paddle.device.cuda.memory_reserved(0) / (1024**3)
print(f"memory_reserved : {cuda_mem_after_used7:.3f} GiB")
