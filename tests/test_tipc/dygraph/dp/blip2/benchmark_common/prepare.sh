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
rm -rf coco.tar.gz
rm -rf /root/.paddlemix/datasets/coco
# dataset
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/blip2/coco.tar.gz
tar -zxvf coco.tar.gz
mv coco /root/.paddlemix/datasets/
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip install einops -i https://mirror.baidu.com/pypi/simple
python -m pip install -r ../requirements.txt
python -m pip install -e ../
python -m pip install --upgrade paddlenlp pybind11 regex sentencepiece tqdm visualdl attrdict easydict pyyaml -i https://mirror.baidu.com/pypi/simple
pip install -r ../paddlemix/appflow/requirements.txt
pip install -U ppdiffusers
python -m pip install ../../paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
python -m pip install paddlenlp==3.0.0b0
python -m pip install huggingface_hub==0.23.0
python -m pip list
cd -
