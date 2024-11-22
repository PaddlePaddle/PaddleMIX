# CogVideoX视频生成

```shell
python infer.py \
  --prompt "a bear is walking in a zoon" \
  --model_path THUDM/CogVideoX-2b/ \
  --generate_type "t2v" \
  --dtype "float16" \
  --seed 42
```