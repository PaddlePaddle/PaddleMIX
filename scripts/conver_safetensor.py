import paddle
import safetensors.paddle
import numpy as np

new_safetensors = {}
metadata = {"total_size": "16582751232",}
layer11_gate_weight = None
layer11_up_weight = None
for idx in range(1, 6):
    file_path = "/path/to/Qwen2-VL-7B-Instruct/model-0000" + str(idx) + "-of-00005.safetensors"
    new_file_path="/new_path/to/Qwen2-VL-7B-Instruct-fuse_qkv/model-0000" + str(idx) + "-of-00005.safetensors"
    theta = (
            safetensors.paddle.load_file(file_path)
            )
    for key, val in theta.items():
        # print("key = ", key, " val.shape = ", val.shape)
        if len(key.split('.')) == 6 and key.split('.', 4)[4] == 'q_proj.weight':
            q_weight = val
            k_weight = theta[key.replace('q_proj', 'k_proj')]
            v_weight = theta[key.replace('q_proj', 'v_proj')]
            qkv_weight = paddle.concat([q_weight, k_weight, v_weight], axis=-1)
            # print(qkv_weight.shape)
            new_safetensors[key.replace('q_proj', 'qkv_proj')] = qkv_weight
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'q_proj.bias':
            q_bias = val
            k_bias = theta[key.replace('q_proj', 'k_proj')]
            v_bias = theta[key.replace('q_proj', 'v_proj')]
            qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)
            # print(qkv_bias.shape)
            new_safetensors[key.replace('q_proj', 'qkv_proj')] = qkv_bias
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'k_proj.weight':
            continue
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'k_proj.bias':
            continue
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'v_proj.weight':
            continue
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'v_proj.bias':
            continue
        elif len(key.split('.')) == 6 and key.split('.', 2)[2] == '11.mlp.up_proj.weight':
            layer11_up_weight = val
        elif len(key.split('.')) == 6 and key.split('.', 2)[2] == '11.mlp.gate_proj.weight':
            layer11_gate_weight = val
            gate_up_weight = paddle.concat([layer11_gate_weight, layer11_up_weight], axis=-1)
            new_safetensors[key.replace('gate_proj', 'gate_up_fused_proj')] = gate_up_weight
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'gate_proj.weight':
            gate_weight = val
            up_weight = theta[key.replace('gate_proj', 'up_proj')]
            gate_up_weight = paddle.concat([gate_weight, up_weight], axis=-1)
            new_safetensors[key.replace('gate_proj', 'gate_up_fused_proj')] = gate_up_weight
        elif len(key.split('.')) == 6 and key.split('.', 4)[4] == 'up_proj.weight':
            continue
        else:
            new_safetensors[key] = val
    # save new safetensors
    safetensors.paddle.save_file(new_safetensors, new_file_path, metadata=metadata)
    print("save new safetensors for ", new_file_path)
    new_safetensors.clear()



