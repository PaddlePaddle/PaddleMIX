#!/usr/bin/env python3
from reprod_log import ReprodDiffHelper

diff_helper = ReprodDiffHelper()
torch_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/pretransform/torch_pretransform.npy")
paddle_info = diff_helper.load_info("/home/onion/workspace/code/pp/Alignment/pretransform/paddle_pretransform.npy")

# compare result and produce log
diff_helper.compare_info(torch_info, paddle_info)
diff_helper.report(
    path="/home/onion/workspace/code/pp/Alignment/pretransform/pretransform_diff.log", diff_threshold=1e-5)
