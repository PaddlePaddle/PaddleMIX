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
# uninstall ppdiffusers and install develop paddlemix
python -m pip uninstall -y ppdiffusers
cd ../ppdiffusers/
python -m pip install -e .
# 安装 stable diffusion 依赖
cd examples/stable_diffusion
pip install -r requirements.txt
pip install soundfile==0.12.1 #20250103版本升级后报错
python -m pip uninstall paddlenlp -y
python -m pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
python -m pip list
# 删除当前目录下的data
# cd ppdiffusers/examples/stable_diffusion
rm -rf data
rm -rf laion400m_demo_data.tar.gz
# 下载 laion400m_demo 数据集
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
# 解压
tar -zxvf laion400m_demo_data.tar.gz

