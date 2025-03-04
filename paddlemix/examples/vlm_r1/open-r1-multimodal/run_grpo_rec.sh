export DEBUG_MODE="true"
export WANDB_DISABLED="true"
export CUDA_VISIBLE_DEVICES=1
export LOG_PATH="./debug_v2.txt"
RUN_NAME="Qwen2.5-VL-3B-GRPO-REC"
IMAGE_ROOT="coco"

python -m paddle.distributed.launch \
    --nnodes=1 \
    --rank=0 \
    --master=127.0.0.1 \
    --nproc_per_node=1 \
    paddlemix/examples/vlm_r1/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name paddlemix/examples/vlm_r1/open-r1-multimodal/data_config/rec.yaml \
    --image_root $IMAGE_ROOT \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --num_generations 8 \
    --fp16_opt_level "O2" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --recompute \
    --logging_steps 1 \
    --bf16 \
    --seed 42 \
    --report_to tensorboard \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 