@@ -1,8 +1,16 @@
环境准备
conda create -n internlm python=3.11 -y
conda activate internlm

git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX
pip install -e .

#ppdiffusers 安装
cd ppdiffusers
pip install -e .

TRUST_REMOTE_CODE=1 python ~/chat_demo.py \
  --model_name_or_path "/home/aistudio/internlm-xcomposer2d5-7b" \
  --image_path "/home/aistudio/internlm-xcomposer2d5-7b/logo_en.png" \
  --text "Please describe this image in detail."
