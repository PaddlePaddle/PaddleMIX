import torch
import paddle
import gc
import tracemalloc
from transformers import AutoModel, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

local_model_path = "/home/aistudio/internlm-xcomposer2d5-7b"
save_dir = "/home/aistudio/internlm-xcomposer2d5-7b-paddle"

# 使用transformers加载PyTorch模型
def load_torch_model(path):
    return AutoModel.from_pretrained(path, trust_remote_code=True)

# 将PyTorch模型参数转换为PaddlePaddle格式
def convert_state_dict(torch_state_dict):
    paddle_state_dict = {}
    for key, tensor in torch_state_dict.items():
        # 确保张量不在meta设备上
        if tensor.device.type == "meta":
            print(f"Warning: Tensor {key} is on meta device and will be skipped.")
            continue
        # 确保张量在CPU上
        tensor_cpu = tensor.cpu()
        p_tensor = paddle.to_tensor(tensor_cpu.numpy())
        if 'weight' in key and any(x in key for x in ['emb', 'linear', 'qkv']):
            p_tensor = p_tensor.transpose([1, 0])
        paddle_state_dict[key] = p_tensor
    return paddle_state_dict

# 保存PaddlePaddle模型参数
def save_paddle_model(paddle_state_dict, save_dir):
    paddle.save(paddle_state_dict, f"{save_dir}/model_state.pdparams")

# 加载模型并分发到不同设备
def load_and_dispatch_model(model_path):
    with init_empty_weights():
        torch_model = load_torch_model(model_path)
    torch_model = load_checkpoint_and_dispatch(
        torch_model,
        model_path,
        device_map="auto",
        offload_folder="/tmp/offload",
        no_split_module_classes=["InternLMXComposer2ForCausalLM"],
        offload_buffers=True,
    )
    return torch_model

# 分阶段处理模型的不同部分
def process_model_in_chunks():
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    
    # 加载并分发模型
    torch_model = load_and_dispatch_model(local_model_path)
    
    # 分阶段处理模型参数
    state_dict = torch_model.state_dict()
    total_items = len(state_dict)
    chunk_size = 100  # 根据需要调整
    for i in range(0, total_items, chunk_size):
        chunk = {k: state_dict[k] for k in list(state_dict.keys())[i:i+chunk_size]}
        paddle_chunk = convert_state_dict(chunk)
        # 保存当前chunk的参数（如果需要）
        # save_paddle_model(paddle_chunk, save_dir)
        # 清理不再需要的变量
        del chunk, paddle_chunk
        gc.collect()
    
    # 保存完整的模型参数
    paddle_state_dict = convert_state_dict(state_dict)
    save_paddle_model(paddle_state_dict, save_dir)
    
    # 清理不再需要的变量
    del torch_model, state_dict, paddle_state_dict
    gc.collect()

# 检测内存泄漏
def detect_memory_leaks():
    tracemalloc.start()
    process_model_in_chunks()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

# 执行模型转换
detect_memory_leaks()

print(f"模型参数已保存到: {save_dir}")