export PYTHONPATH=/root/project/paddlenlp/lvdm/paddle/PaddleMIX/ppdiffusers:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
set -eux

python lvdm_sample_short.py
