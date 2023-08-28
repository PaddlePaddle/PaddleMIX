rm -rf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
rm -rf CompVis-stable-diffusion-v1-4-paddle-init
rm -rf laion400m_demo_data.tar.gz
rm -rf data

wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
tar -zxvf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz


export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip install einops -i https://mirror.baidu.com/pypi/simple
python -m pip install -r ../requirements.txt
python -m pip install --upgrade paddlenlp pybind11 regex sentencepiece tqdm visualdl attrdict easydict pyyaml -i https://mirror.baidu.com/pypi/simple

# uninstall ppdiffusers and install develop paddlemix
python -m pip uninstall -y ppdiffusers
python -m pip install -e ../
python -m pip list
cd -
