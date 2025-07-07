# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from pathlib import Path

import paddle

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.misc import enumerate_chunk
from paddlemix.datacopilot.nn import PPInsCapTagger


class QAschema(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert len(values) % 2 == 0, "QA content must be a list of pairs"
        values = list(zip(values[0::2], values[1::2]))
        setattr(namespace, self.dest, values)


if __name__ == "__main__":

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-m", "--model-name-or-path", type=str, default="paddlemix/PP-InsCapTagger")
    base.add_argument("-t", "--dtype", type=str, default="float16")
    base.add_argument("-k", "--k-start", type=int, default=0)
    base.add_argument("-o", "--output-dir", default="SFT_tag_output_test")
    base.add_argument("--seed", type=int, default=0)

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(help="mod of data: json_data/single_data", dest="mod")
    json_parser = subs.add_parser("json_data", parents=[base])
    json_parser.add_argument("-d", "--dataset-path", type=str, required=True)

    single_parser = subs.add_parser("single_data", parents=[base])
    single_parser.add_argument("-image", "--image-path", type=str, required=True)
    single_parser.add_argument("-qa", "--qa-content", nargs="+", type=str, required=True, action=QAschema)

    args = parser.parse_args()
    paddle.seed(seed=args.seed)

    if args.mod == "json_data":
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        m = PPInsCapTagger(args.model_name_or_path)
        dataset = MMDataset.from_auto(args.dataset_path)
        print("loading dataset...")
        print("data size==", len(dataset))
        for i, subdata in enumerate_chunk(dataset, chunk_size=1000, start=args.k_start):
            print(f"convert {i}th(1000) data")
            subdata: MMDataset
            subdata = subdata.map(m.inference, max_workers=1)
            subdata.export_json(f"{args.output_dir}/tagger_{i:05}.json")
            print(f"{i*1000}th(1000) data save to {args.output_dir}/tagger_{i:05}.json")

    if args.mod == "single_data":
        item = {}
        item["image"] = args.image_path
        item["conversations"] = args.qa_content
        m = PPInsCapTagger(args.model_name_or_path)
        tag_item = m(item)
        print(tag_item)
