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
from functools import partial

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.nn.llms import ErnieEval
from paddlemix.datacopilot.nn.layout_parsing import PaddleXLayoutParser
from paddlemix.datacopilot.ops.generate.pp_infinity_doc import PPInfinityDocData


def generate_for_single_image(item: dict, generator: PPInfinityDocData):    
    conversations = generator.generate_doc(item['layout'])
    return item.update({'conversations': conversations})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, )
    parser.add_argument('--output_json_path', type=str, required=True, )
    parser.add_argument('--gpu_id', type=int, default=0, )
    parser.add_argument('--num_gpus', type=int, default=1, )
    parser.add_argument('--processes_per_gpu', type=int, default=1, )

    args = parser.parse_args()

    generator = PPInfinityDocData(llm=ErnieEval())
    layout_parser = PaddleXLayoutParser(gpu_id=args.gpu_id)
    
    paths = list(sorted(Path(args.root).glob('*.jpg')))

    layouts = layout_parser.process_images(paths, num_gpus=args.num_gpus, processes_per_gpu=args.processes_per_gpu)
    items = [dict(image=path, layout=layout) for path, layout in zip(paths, layouts) if layout]

    (
        MMDataset(items)
        .map(partial(generate_for_single_image, generator=generator))
        .nonempty()
        .export_json(args.output_json_path)
    )



