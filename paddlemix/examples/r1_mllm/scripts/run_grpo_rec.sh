# export DEBUG_MODE="true"
# export LOG_PATH="./debug_grpo_rec_sample.txt"
# export CUDA_VISIBLE_DEVICES=0,2

GPUS=${GPUS:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
RUN_NAME="Qwen2.5-VL-3B-GRPO-REC_${GPUS}GPUS_stage2"
IMAGE_ROOT="data/coco"

python -m paddle.distributed.launch \
    --nnodes=1 \
    --master=127.0.0.1 \
    --nproc_per_node=$GPUS \
    paddlemix/examples/r1_mllm/train/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name paddlemix/examples/r1_mllm/data_config/rec.yaml \
    --image_root $IMAGE_ROOT \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --num_generations $NUM_GENERATIONS \
    --fp16_opt_level "O2" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --num_train_epochs 1 \
    --ignore_save_lr_and_optim True \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --do_train \
    --sharding "stage2" \
    --amp_master_grad=1 \
    --hybrid_parallel_topo_order="sharding_first" \
    --attn_implementation "eager" \
    --freeze_vision False \
    --max_steps 500 