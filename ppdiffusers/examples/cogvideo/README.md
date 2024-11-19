# CogVideoX视频生成

```shell
python infer.py \
  --prompt "a bear is walking in a zoon" \
  --model_path paddle_weights/THUDM/CogVideoX-2b/ \
  --generate_type "t2v" \
  --dtype "bfloat16" \
  --seed 42
```