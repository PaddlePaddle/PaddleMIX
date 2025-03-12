GPUS=${GPUS:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
RUN_NAME="Qwen2.5-VL-3B-GRPO-Geometry_${GPUS}"
IMAGE_ROOT="data/GEOQA_R1V_Train_8K"
DATASET_NAME="data/GEOQA_R1V_Train_8K"

python -m paddle.distributed.launch \
    --nnodes=1 \
    --rank=0 \
    --master=127.0.0.1 \
    --nproc_per_node=$GPUS \
    paddlemix/examples/r1_mllm/train/grpo_r1-v.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name $DATASET_NAME \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --num_generations $NUM_GENERATIONS \
    --fp16_opt_level "O2" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --seed 42 \
    --report_to tensorboard \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --sharding="stage2" \
    --amp_master_grad True \
    --do_train \
    --ignore_save_lr_and_optim True \
    --freeze_vision True
    # --recompute \

