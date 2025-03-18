export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=163840
export FLAGS_mla_use_tensorcore=1

python deploy/deepseek_vl2/deepseek_vl2_infer.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --mla_use_matrix_absorption \
    --benchmark