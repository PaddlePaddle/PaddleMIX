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

rm -rf llava_bench_data.tar
rm -rf /root/.paddlemix/datasets/llava_bench_data
# dataset
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/llava_bench_data.tar
tar -xf llava_bench_data.tar
mv llava_bench_data /root/.paddlemix/datasets/
rm -rf llava_bench_data.tar
ln -s /root/.paddlemix/datasets/llava_bench_data ./

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip install einops -i https://mirror.baidu.com/pypi/simple
python -m pip install -r ../requirements.txt
python -m pip install -e ../
python -m pip install --upgrade paddlenlp pybind11 regex sentencepiece tqdm visualdl attrdict easydict pyyaml -i https://mirror.baidu.com/pypi/simple
pip install -r ../paddlemix/appflow/requirements.txt
pip install -U ppdiffusers
bash ../build_paddle_env.sh
# python -m pip install https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post118-cp310-cp310-linux_x86_64.whl
python -m pip install paddlenlp==3.0.0b2
python -m pip install huggingface_hub==0.23.0
python -m pip list
cd -
