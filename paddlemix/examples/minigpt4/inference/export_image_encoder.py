import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
os.environ["FLAGS_use_cuda_managed_memory"]="true"

import paddle
from paddlemix import MiniGPT4ForConditionalGeneration


def export(args):
    model = MiniGPT4ForConditionalGeneration.from_pretrained(args.minigpt4_13b_path, vit_dtype="float16")
    model.eval()

    # convert to static graph with specific input description
    model = paddle.jit.to_static(
        model.encode_images,
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
        "--minigpt4_13b_path",
        default="your minigpt4 dir path",
        type=str,
        help="The dir name of minigpt4 checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        default="./checkpoints/encode_image/encode_image",
        type=str,
        help="The saving path of static minigpt4.",
    )
    args = parser.parse_args()

    export(args)








# processor = MiniGPT4Processor.from_pretrained(minigpt4_13b_path)
# print("load processor and model done!")

# # prepare model inputs for MiniGPT4
# url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor.process_images(image)
# model.


# # generate with MiniGPT4
# outputs = model.generate(**inputs, **generate_kwargs)
# msg = processor.batch_decode(outputs[0])
# print(msg)
