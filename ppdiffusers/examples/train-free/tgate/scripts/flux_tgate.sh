which python

CUDA_VISIBLE_DEVICES=3 python main.py \
--prompt "A cat holding a sign that says hello world" \
--model 'flux' \
--gate_step 25 \
--sp_interval 2 \
--fi_interval 1 \
--warm_up 2 \
--saved_path './generated_tmp/flux/' \
--inference_step 50 \
--seed 42