# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import base64
import gzip
import json
import multiprocessing
import os
from pathlib import Path

from fastcore.all import chunked

file_abs_path = Path(os.path.abspath(__file__)).parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to you raw jsonl files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output part file.")
    parser.add_argument(
        "--output_name",
        type=str,
        default="custom_dataset",
        help="The name of output dataset. Defaults to `custom_dataset`.",
    )
    parser.add_argument(
        "--caption_key", type=str, default="caption", help="The caption key of json file. Defaults to `caption`."
    )
    parser.add_argument(
        "--image_key", type=str, default="image", help="The image key of json file. Defaults to `image`."
    )
    parser.add_argument(
        "--per_part_file_num",
        type=int,
        default=1000,
        help="The number of files contained in each part file. Defaults to `1000`.",
    )
    parser.add_argument(
        "--save_gzip_file", action="store_true", help="Whether to save gzip file. Defaults to `False`."
    )
    parser.add_argument("--num_repeat", type=int, default=1, help="Defaults to `1`.")
    args = parser.parse_args()
    return args


def load_jsonl(filename, per_part_file_num=1000):
    outputs = []
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            try:
                outputs.append(json.loads(line))
            except Exception as e:
                print(e)
    return chunked(outputs, chunk_sz=per_part_file_num)


def data_to_part(data_list, output_file, base_dir, save_gzip_file=True, image_key="image", caption_key="caption"):
    len_files = 0
    fileopen = gzip.open if save_gzip_file else open

    with fileopen(output_file, "wb") as fout:
        for line in data_list:
            capiton = line[caption_key]
            image_path = os.path.join(base_dir, line[image_key])
            if os.path.exists(image_path):
                with open(image_path, "rb") as im_file:
                    encoded_string = base64.b64encode(im_file.read()).decode()
                out_bytes = "\t".join([capiton, "NONE", encoded_string]).encode("utf-8")
                fout.write(out_bytes)
                fout.write(b"\n")
                len_files += 1
            else:
                print("Image not found: ", image_path)
    print(f"Process {output_file} done, It has {len_files} files. ")
    return len_files, output_file


def main(args):
    data_chunk_list = load_jsonl(args.input_path, args.per_part_file_num)
    output_path = Path(args.output_path)
    name = Path(args.input_path).stem if args.output_name is None else args.output_name
    output_path = output_path / "laion400m_format_data"
    output_path.mkdir(exist_ok=True, parents=True)

    # create filelist dir
    filelist_dir = output_path.parent / "filelist"
    filelist_dir.mkdir(exist_ok=True, parents=True)

    # create filelist and filelist.list
    filelist_path = filelist_dir / f"{name}.filelist"
    filelistlist_path = filelist_dir / f"{name}.filelist.list"

    base_dir = Path(os.path.abspath(args.input_path)).parent
    jobs = []
    filelist_data = []
    for index, data_chunk in enumerate(data_chunk_list, start=1):
        data_name = "part-{:06d}".format(index)
        if args.save_gzip_file:
            data_name += ".gz"
        output_file = os.path.join(str(output_path), data_name)
        p = multiprocessing.Process(target=data_to_part, args=(data_chunk, output_file, base_dir, args.save_gzip_file))
        jobs.append(p)
        p.start()
        filelist_data.append(str(Path(output_file).absolute().relative_to(file_abs_path)))

    filelist_data = filelist_data * args.num_repeat
    filelist_path.write_text("\n".join(filelist_data), encoding="utf-8")
    filelistlist_path.write_text(str(filelist_path.absolute().relative_to(file_abs_path)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    main(args)
