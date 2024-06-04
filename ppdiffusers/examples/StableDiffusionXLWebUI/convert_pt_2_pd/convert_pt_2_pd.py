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
import os

import paddle
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description="Convert pytorch model safetensor weight to pdparams.")
    parser.add_argument("--model_name", type=str, help="Your src model path")
    args = parser.parse_args()
    # 待转换pytorch模型路径在此修改
    model_torch_name = args.model_name

    model_component_name = []
    for r, d, f in os.walk(model_torch_name):
        model_component_name.extend([os.path.join(r, fn) for fn in f if fn.endswith(".safetensors")])

    # 目标模型路径
    model_dir = "./" + model_torch_name.split("/")[-1].title()
    os.makedirs(model_dir, exist_ok=True)
    # build pipeline components
    os.system(f"cp -r ./convert_pt_2_pd/basemodel/* {model_dir}")

    count = 0
    for model_name in model_component_name:
        f2 = safe_open(model_name, framework="np")

        # 保存键的信息
        with open(f'{model_dir}/{model_name.split("/")[-2]}.txt', "w") as f:
            try:
                # 现在print()的输出将被写入到文件中
                for k in f2.keys():  # 查看key
                    print(k, file=f)
            except Exception as e:
                print(e, file=f)

        state_dict = {}
        try:
            for k in f2.keys():
                v = f2.get_tensor(k)
                if len(v.shape) == 2 and not ("position_embedding" in k or "token_embedding" in k):
                    state_dict[k] = paddle.to_tensor(v, dtype=v.dtype).t()
                elif "att" in k and len(v.shape) == 4:
                    state_dict[k] = paddle.to_tensor(v.squeeze(), dtype=v.dtype).t()
                else:
                    state_dict[k] = paddle.to_tensor(v, dtype=v.dtype)
            paddle.save(state_dict, f'{model_dir}/{model_name.split("/")[-2]}/model_state.pdparams')
            paddle.device.cuda.empty_cache()
            count += 1
        except Exception as e:
            print("错误原因：", e)

    if count == 4:
        os.system(f"rm -rf {model_torch_name}")
        print("转换完成")


if __name__ == "__main__":
    main()
