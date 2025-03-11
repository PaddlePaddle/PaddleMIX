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

rm -rf playground.tar
rm -rf /root/.paddlemix/datasets/playground
mkdir -p /root/.paddlemix/datasets/
# dataset
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/playground.tar
tar -xf playground.tar
mv playground /root/.paddlemix/datasets/
rm -rf playground.tar
ln -s /root/.paddlemix/datasets/playground ./

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install -r ../requirements.txt
python -m pip install -e ../
#  python -m pip install paddlepaddle-gpu==3.0.0b2 #-i https://www.paddlepaddle.org.cn/packages/stable/cu123/
python -m pip install paddlenlp==3.0.0b3
python -m pip list
cd -
