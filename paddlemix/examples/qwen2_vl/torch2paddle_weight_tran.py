# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
import shutil

import paddle
import torch
from paddlenlp.utils.log import logger
from safetensors.numpy import save_file
from safetensors.torch import load_file

NEED_TRANSPOSE_KEYS = {
    "2B": {
        "up_proj.weight",
        "gate_proj.weight",
        "down_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "attn.proj.weight",
        "attn.qkv.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
        "merger.mlp.0.weight",
        "merger.mlp.2.weight",
    },
    "7B": {
        "up_proj.weight",
        "gate_proj.weight",
        "down_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "lm_head.weight",  # 7B
        "attn.proj.weight",
        "attn.qkv.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
        "merger.mlp.0.weight",
        "merger.mlp.2.weight",
    },
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Convert PyTorch safetensors model to PaddlePaddle format.")
    parser.add_argument(
        "--model_size", type=str, default="2B", choices=["2B", "7B"], help="Model size, can be 2B or 7B"
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="torch_models/Qwen2-VL-2B-Instruct",
        help="Path to source PyTorch model directory",
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        default="paddle_models/Qwen2-VL-2B-Instruct",
        help="Path to destination PaddlePaddle model directory",
    )

    parser.add_argument("--src_prefix", type=str, default="xxxxx.", help="Source key prefix to replace")

    parser.add_argument("--dst_prefix", type=str, default="xxxxx.", help="Destination key prefix")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination directory if it exists")

    parser.add_argument("--skip_config_update", action="store_true", help="Skip updating config files")

    return parser.parse_args()


def check_if_transpose_needed(key: str, model_size: str) -> bool:
    """Check if the given tensor key needs transpose operation.

    Args:
        key: The tensor key to check

    Returns:
        bool: True if transpose is needed, False otherwise
    """
    return any(x in key for x in NEED_TRANSPOSE_KEYS[model_size])


def convert_tensor_to_paddle(tensor: torch.Tensor, needs_transpose: bool) -> paddle.Tensor:
    """Convert PyTorch tensor to PaddlePaddle tensor with optional transpose.

    Args:
        tensor: Input PyTorch tensor
        needs_transpose: Whether to transpose the tensor

    Returns:
        Converted PaddlePaddle tensor
    """
    if needs_transpose:
        tensor = tensor.cpu().t().contiguous()
    else:
        tensor = tensor.cpu()

    capsule = torch.utils.dlpack.to_dlpack(tensor)
    return paddle.utils.dlpack.from_dlpack(capsule)


def process_safetensors_file(args, src_file_path: str, dest_file_path: str, src_prefix: str, dst_prefix: str) -> None:
    """Process a single safetensors file, converting PyTorch tensors to PaddlePaddle format.

    Args:
        src_file_path: Path to source safetensors file
        dest_file_path: Path to destination safetensors file
        src_prefix: Source key prefix to replace
        dst_prefix: Destination key prefix
    """
    tensors = load_file(src_file_path)
    processed_tensors = {}

    for key, tensor in tensors.items():
        dest_key = key.replace(src_prefix, dst_prefix)
        needs_transpose = check_if_transpose_needed(key, model_size=args.model_size)

        logger.info("Processing key: {}, shape: {}, transpose: {}".format(key, tensor.shape, needs_transpose))

        paddle_tensor = convert_tensor_to_paddle(tensor, needs_transpose)
        processed_tensors[dest_key] = paddle_tensor.numpy()

        logger.info("Converted to key: {}, shape: {}".format(dest_key, processed_tensors[dest_key].shape))

    save_file(processed_tensors, dest_file_path, metadata={"format": "np"})


def update_config_files(dest_dir: str) -> None:
    """Update configuration files in the destination directory.

    Args:
        dest_dir: Path to destination directory
    """
    config_files = ["config.json", "generation_config.json"]

    for file_name in config_files:
        file_path = os.path.join(dest_dir, file_name)
        if not os.path.exists(file_path):
            continue

        # Update torch_dtype to dtype
        os.system(f"sed -i -e 's/torch_dtype/dtype/g' {file_path}")
        # Remove transformers_version line
        os.system(f"sed -i /transformers_version/d {file_path}")

        if file_name == "generation_config.json":
            # Remove comma after max_new_tokens
            os.system(f"sed -i '/max_new_tokens/s/,//g' {file_path}")


def process_model_directory(
    args, src_dir: str, dest_dir: str, src_prefix: str, dst_prefix: str, skip_config_update: bool = False
) -> None:
    """Process the entire model directory.

    Args:
        src_dir: Path to source model directory
        dest_dir: Path to destination model directory
        src_prefix: Source key prefix to replace
        dst_prefix: Destination key prefix
        skip_config_update: Whether to skip config file updates
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if os.path.exists(dest_dir):
        logger.warning(f"Destination directory already exists: {dest_dir}")
        if not args.overwrite:
            raise FileExistsError(f"Destination directory exists and --overwrite not specified: {dest_dir}")
        logger.info("Cleaning existing destination directory")
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)

    index_file = os.path.join(src_dir, "model.safetensors.index.json")

    if os.path.exists(index_file):
        # Process indexed safetensors
        with open(index_file, "r") as f:
            index = json.load(f)

        dst_index = copy.deepcopy(index)
        # Update weight map keys
        dst_index["weight_map"] = {k.replace(src_prefix, dst_prefix): v for k, v in dst_index["weight_map"].items()}

        files_to_process = set(index["weight_map"].values())
        logger.info("Files to process: {}".format(files_to_process))

        for file_name in sorted(os.listdir(src_dir)):
            if file_name.startswith(".") or file_name.startswith("chat_template."):
                continue

            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)

            if file_name in files_to_process:
                logger.info(f"Processing safetensors file: {file_name}")
                process_safetensors_file(args, src_path, dest_path, src_prefix, dst_prefix)
            else:
                logger.info(f"Copying non-tensor file: {file_name}")
                shutil.copy(src_path, dest_path)

        # Save updated index file
        with open(os.path.join(dest_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(dst_index, f, indent=2)
    else:
        # Process single safetensors file
        for file_name in sorted(os.listdir(src_dir)):
            if file_name.startswith(".") or file_name.startswith("chat_template."):
                continue

            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)

            if file_name == "model.safetensors":
                logger.info("Processing main safetensors file")
                process_safetensors_file(src_path, dest_path, src_prefix, dst_prefix)
            else:
                logger.info(f"Copying non-tensor file: {file_name}")
                shutil.copy(src_path, dest_path)

    # Update config files unless skipped
    if not skip_config_update:
        update_config_files(dest_dir)
    else:
        logger.info("Skipping config file updates as requested")


def main() -> None:
    """Main function to execute the conversion process."""
    args = parse_arguments()

    # Set default destination directory if not specified
    dest_dir = args.dest_dir

    logger.info("Starting model conversion")
    logger.info(f"Source path: {args.src_dir}")
    logger.info(f"Destination path: {dest_dir}")
    logger.info(f"Source prefix: {args.src_prefix}")
    logger.info(f"Destination prefix: {args.dst_prefix}")

    try:
        process_model_directory(
            args=args,
            src_dir=args.src_dir,
            dest_dir=dest_dir,
            src_prefix=args.src_prefix,
            dst_prefix=args.dst_prefix,
            skip_config_update=args.skip_config_update,
        )
        logger.info("Model conversion completed successfully")
    except Exception as e:
        logger.error(f"Error during model conversion: {str(e)}")
        raise


if __name__ == "__main__":
    main()
