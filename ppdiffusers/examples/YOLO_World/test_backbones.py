#!/usr/bin/env python3
from reprod_log import ReprodDiffHelper

diff_helper = ReprodDiffHelper()
torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/backbones/torch_textfeats.npy")
paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/backbones/paddle_textfeats.npy")
# torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/backbones/torch_imgfeats1.npy")
# paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/backbones/paddle_imgfeats1.npy")

# compare result and produce log
diff_helper.compare_info(torch_info, paddle_info)
diff_helper.report(
    path="/home/onion/workspace/code/pp/Alignment/backbones/backbone_text_diff.log", diff_threshold=1e-5)
# diff_helper.report(
#     path="/home/onion/workspace/code/pp/Alignment/backbones/backbone_img1_diff.log", diff_threshold=1e-5)
