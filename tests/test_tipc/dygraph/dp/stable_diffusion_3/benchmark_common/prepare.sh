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


if [ ! -d "stable-diffusion-3-medium-diffusers-paddle-init" ]; then
    echo "Downloading stable-diffusion-3-medium-diffusers-paddle-init.tar.gz..."
    wget https://bj.bcebos.com/paddlenlp/models/community/westfish/sd3_benchmark/stable-diffusion-3-medium-diffusers-paddle-init.tar.gz
    echo "Extracting stable-diffusion-3-medium-diffusers-paddle-init.tar.gz..."
    tar -zxvf stable-diffusion-3-medium-diffusers-paddle-init.tar.gz
else
    echo "Directory stable-diffusion-3-medium-diffusers-paddle-init already exists. Skipping download."
fi

if [ ! -d "dog" ]; then
    echo "Downloading dog.zip..."
    wget https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip
    echo "Unzipping dog.zip..."
    unzip dog.zip
else
    echo "Directory dog already exists. Skipping download."
fi


RUN_SETUP=${RUN_SETUP:-"true"}
if [ "$RUN_SETUP" = "true" ]; then
    echo "Running setup and installation steps..."

    export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
    python -m pip install --upgrade pip
    # python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    python -m pip install einops
    python -m pip install -r ../requirements.txt
    python -m pip install --upgrade pybind11 regex sentencepiece tqdm visualdl attrdict easydict pyyaml paddlesde
    python -m pip install huggingface-hub==0.23.0

    # uninstall ppdiffusers and install develop paddlemix
    python -m pip uninstall -y ppdiffusers
    cd ../ppdiffusers/
    python -m pip install -e .
    cd -
    cd ../ppdiffusers/examples/dreambooth
    pip install -r requirements_sd3.txt
    cd -
    python -m pip uninstall paddlenlp -y
    python -m pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

    python -m pip list
else
    echo "fast mode, skipping setup and installation steps as RUN_SETUP is set to false."
fi
