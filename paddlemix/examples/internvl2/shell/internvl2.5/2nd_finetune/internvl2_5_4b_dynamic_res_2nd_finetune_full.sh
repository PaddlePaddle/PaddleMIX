set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
tensor_parallel_degree=${tensor_parallel_degree:-1}
sharding_parallel_degree=$((GPUS / tensor_parallel_degree))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_chat_v2_5/internvl2_5_4b_dynamic_res_2nd_finetune_full'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node ${GPUS} --rank 0 --ips ${TRAINER_INSTANCES} --run_mode=collective"
${TRAINING_PYTHON} --log_dir ${OUTPUT_DIR}/paddle_distributed_logs \
  paddlemix/examples/internvl2/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/InternVL2_5-4B" \
  --conv_style "internvl2_5" \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${OUTPUT_DIR}/logs \
  --meta_path "paddlemix/examples/internvl2/shell/data/internvl_1_2_finetune.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --fp16 False \
  --fp16_opt_level "O1" \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --recompute True \
  --max_grad_norm 1.0 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --optim "adamw" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "visualdl" \
  --tensor_parallel_degree=${tensor_parallel_degree} \
  --sharding_parallel_degree=${sharding_parallel_degree} \
  --pipeline_parallel_degree=1 \
  --sep_parallel_degree=1 \
  --sharding="stage2" \
  --amp_master_grad=1 \
  --hybrid_parallel_topo_order="sharding_first" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
