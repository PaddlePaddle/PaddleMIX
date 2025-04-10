import argparse

import paddle
from paddlenlp.peft import LoRAConfig, LoRAModel

from paddlenlp.transformers import DeepseekTokenizerFast
from paddlemix.models.deepseek_vl2 import DeepseekVLV2ForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of pretrained model.")
    parser.add_argument(
        "--lora_path", default=None, required=True, help="The directory of LoRA parameters. Default to None"
    )
    parser.add_argument("--merge_model_path", default=None, help="The directory of merged parameters. Default to None")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    return parser.parse_args()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)
    lora_config = LoRAConfig.from_pretrained(args.lora_path)
    dtype = lora_config.dtype
    lora_config.merge_weights = True

    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
    )
    model = LoRAModel.from_pretrained(model=model, lora_path=args.lora_path, lora_config=lora_config)
    model.eval()
    if args.merge_model_path is None:
        args.merge_model_path = args.lora_path

    model_state_dict = model.model.state_dict()
    for key in list(model_state_dict):
        if "lora" in key:
            del model_state_dict[key]
    model.model.save_pretrained(args.merge_model_path, state_dict=model_state_dict,safe_serialization=True)

    # save tokenizer config
    tokenizer = DeepseekTokenizerFast.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(args.merge_model_path)


if __name__ == "__main__":
    merge()