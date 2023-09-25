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
set -x

pip install -r requirements.txt
pip uninstall ppdiffusers -y
cd ../../
pip install -e .
cd -

python download_sd15.py

cd ../../examples/dreambooth
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
cd -

cd ../../examples/textual_inversion
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/cat-toy.tar.gz
tar -zxvf cat-toy.tar.gz
cd -

cd ../../examples/text_to_image_laion400m
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
cp laion400m_demo_data.tar.gz ../stable_diffusion
cp laion400m_demo_data.tar.gz ../autoencoder/vae
cd -

cd ../../examples/stable_diffusion
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
cd -

cd ../../examples/autoencoder/vae
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
cd -

cd ../../examples/controlnet
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/fill50k.zip
unzip -o fill50k.zip
cd -

cd ../../examples/t2i-adapter
wget https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/openpose_data_demo.tar.gz
tar -zxvf openpose_data_demo.tar.gz