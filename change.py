import paddle

state_dict = paddle.load("/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-change/model_state.pdparams")
new_state_dict = {}
for key in state_dict.keys():
    if key.startswith("transformer"):
        new_key = key.replace("transformer", "qwen")
        new_state_dict[new_key] = state_dict[key]
    else:
        new_state_dict[key] = state_dict[key]
paddle.save(new_state_dict, "/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-change/model_state.pdparams")