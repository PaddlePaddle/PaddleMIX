



import paddle

# text_encoder
# state_dict = paddle.load("/home/work/wuxirui/newmmdit10b/output/model/t2p_mmdit/10bgaofanhua/paddle_dynamic/ERNIE-Turbo-2-Chat-0228/model_state.pdparams")

# mmdit
state_dict = paddle.load("/root/.cache/paddlenlp/ppdiffusers/stabilityai/stable-diffusion-3-medium-diffusers/transformer/infer.pdiparams")

# vae
# state_dict = paddle.load("/home/work/wuxirui/newmmdit10b/output/model/t2p_mmdit/10bgaofanhua/paddle_dynamic/vae_encoder8_decoder8_channel16/model_state.pdparams")

# all_keys = list(state_dict.keys)
# print(len(all_keys))
print(state_dict)
# for k in all_keys:
    # state_dict[k] = paddle.cast(state_dict[k], dtype="bfloat16")
    # print(state_dict[k].dtype)


# to_file = "/home/work/wuxirui/newmmdit10b/output/model/t2p_mmdit/10bgaofanhua/paddle_dynamic/generator-1024-206000_bf16.pdparams"
# paddle.save(state_dict, to_file)





