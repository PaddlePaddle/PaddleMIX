#!/usr/bin/env python3
from reprod_log import ReprodDiffHelper

diff_helper = ReprodDiffHelper()

torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/torch_layer0out_attn1_q_input.npy")
paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/paddle_layer0out_attn1_q_input.npy")

diff_helper.compare_info(torch_info, paddle_info)
diff_helper.report(
    path="/home/onion/workspace/code/pp/Alignment/suckclip/clip_attn_input.log", diff_threshold=1e-5)

torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/torch_layer0out_attn1_q_weight.npy")
paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/paddle_layer0out_attn1_q_weight.npy")

diff_helper.compare_info(torch_info, paddle_info)
diff_helper.report(
    path="/home/onion/workspace/code/pp/Alignment/suckclip/clip_weight_diff.log", diff_threshold=1e-5)

torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/torch_layer0out_attn1_q_bias.npy")
paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/paddle_layer0out_attn1_q_bias.npy")

diff_helper.compare_info(torch_info, paddle_info)
diff_helper.report(
    path="/home/onion/workspace/code/pp/Alignment/suckclip/clip_bias_diff.log", diff_threshold=1e-5)

torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/torch_layer0out_attn1_q_output.npy")
paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/suckclip/paddle_layer0out_attn1_q_output.npy")

# compare result and produce log
diff_helper.compare_info(torch_info, paddle_info)
diff_helper.report(
    path="/home/onion/workspace/code/pp/Alignment/suckclip/clip_attn_output.log", diff_threshold=1e-5)
