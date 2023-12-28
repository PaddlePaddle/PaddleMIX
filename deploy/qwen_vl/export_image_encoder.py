import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["FLAGS_use_cuda_managed_memory"]="true"

import paddle
from paddlemix.models.qwen_vl import Vision
from paddlenlp.transformers.configuration_utils import PretrainedConfig

def export(args):
    config = PretrainedConfig.from_pretrained(args.qwen_vl_7b_path)
    model = Vision(config.visual)
    model.eval()

    # convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype="float32"),  # images
        ])

    # save to static model
    paddle.jit.save(model, args.save_path)
    print(f"static model has been to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen_vl_7b_path",
        default="your qwen-vl dir path",
        type=str,
        help="The dir name of qwen-vl checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        default="./checkpoints/encode_image/encode_image",
        type=str,
        help="The saving path of static qwen-vl.",
    )
    args = parser.parse_args()

    export(args)
