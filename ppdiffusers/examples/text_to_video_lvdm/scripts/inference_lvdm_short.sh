ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
set -eux

python lvdm_sample_short.py
