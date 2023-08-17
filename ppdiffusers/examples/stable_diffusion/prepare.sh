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

pip install -r requirements.txt

rm -rf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
rm -rf CompVis-stable-diffusion-v1-4-paddle-init

wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
tar -zxvf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz

rm -rf laion400m_demo_data.tar.gz
rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz
