#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test training benchmark for a model.
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num}
function _set_params(){
    model_item=${1:-"llava-v1_6-vicuna-7b-sft"}   # (必选) 模型 item |llava-v1_6-vicuna-7b-sft|llava-v1_6-vicuna-13b-sft|llava-v1_6-vicuna-7b-pretrain|llava-v1_6-vicuna-7b-lora_sft|llava-v1_6-vicuna-13b-pretrain|llava-v1_6-vicuna-13b-lora_sft
    base_batch_size=${2:-"1"}       # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=${3:-"bf16O2"}            # (必选) fp32|fp16|bf16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递

    model_repo="PaddleMIX"          # (必选) 模型套件的名字
    speed_unit="sample/sec"         # (必选)速度指标单位
    skip_steps=2                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_epochs=${6:-"3"}                 # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${7:-"2"}                # (可选)
    is_large_model=False           # (可选)普通模型默认为False，如果添加大模型且只取一条ips设置为True

    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}

    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs
    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"
    if [ ${profiling} = "true" ];then
            add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
            log_file=${profiling_log_file}
        else
            add_options=""
            log_file=${train_log_file}
    fi

    #模型权重
    if [ ${model_item} = "llava-v1_6-vicuna-7b-sft" ]; then
        model_path="llava-v1.6-vicuna-7b"
        train_stage="sft"
    fi
    if [ ${model_item} = "llava-v1_6-vicuna-7b-pretrain" ]; then
        model_path="vicuna-7b-v1.5"
        train_stage="pretrain"
    fi
    if [ ${model_item} = "llava-v1_6-vicuna-7b-lora_sft" ]; then
        model_path="llava-v1.6-vicuna-7b"
        train_stage="lora_sft"
    fi
    if [ ${model_item} = "llava-v1_6-vicuna-13b-sft" ]; then
        model_path="llava-v1.6-vicuna-13b"
        train_stage="sft"
    fi
    if [ ${model_item} = "llava-v1_6-vicuna-13b-pretrain" ]; then
        model_path="vicuna-13b-v1.5"
        train_stage="pretrain"
    fi
    if [ ${model_item} = "llava-v1_6-vicuna-13b-lora_sft" ]; then
        model_path="llava-v1.6-vicuna-13b"
        train_stage="lora_sft"
    fi



    #训练精度
    if [ ${fp_item} = "fp16O1" ]; then
        use_fp16_cmd="--fp16 True --fp16_opt_level O1"
    fi
    if [ ${fp_item} = "bf16O1" ]; then
        use_fp16_cmd="--bf16 True --fp16_opt_level O1"
    fi

    FUSED=False
    if [ ${fp_item} = "fp16O2" ]; then
        use_fp16_cmd="--fp16 True --fp16_opt_level O2 --amp_master_grad=1"
        FUSED=True
    fi
    if [ ${fp_item} = "bf16O2" ]; then
        use_fp16_cmd="--bf16 True --fp16_opt_level O2 --amp_master_grad=1"
        FUSED=True
    fi
    rm -rf ./work_dirs/${model_item}_benchmark

    export FLAG_USE_EMA=0
    export FLAG_BENCHMARK=1
    export FLAG_RECOMPUTE=1
    export FLAG_XFORMERS=1
    # use fused linear in amp o2 level
    # export FLAG_FUSED_LINEAR=${FUSED}

    # add some flags
    export FLAGS_use_cuda_managed_memory=true
    export FLAGS_allocator_strategy=auto_growth
    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0
    export http_proxy=agent.baidu.com:8188
    export https_proxy=agent.baidu.com:8188

    #训练阶段
    if [ ${train_stage} = "sft" ]; then
        train_cmd="../paddlemix/tools/supervised_finetune.py \
            --do_train true \
            --model_name_or_path liuhaotian/${model_path} \
            --dataset {\"train\":[{\"name\":\"chatml_dataset\",\"data_files\":\"./llava_bench_data/ScienceQA_val_500.json\",\"chat_template\":\"./llava_bench_data/chat_template.json\"}]} \
            --mixtoken false \
            --output_dir ./work_dirs/${model_item}_benchmark \
            --overwrite_output_dir true \
            --logging_steps=1 \
            --recompute true \
            --per_device_train_batch_size ${base_batch_size} \
            --gradient_accumulation_steps 1 \
            --group_by_modality_length false \
            --num_train_epochs ${max_epochs} \
            --learning_rate 2e-05 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --evaluation_strategy no \
            --save_strategy no \
            --max_length 2048 \
            ${use_fp16_cmd} \
            --do_eval false \
            --disable_tqdm true \
            --tensor_parallel_degree=1 \
            --sharding_parallel_degree=8 \
            --sharding="stage2" \
            --skip_memory_metrics false \
            --benchmark true
            "
    fi
    if [ ${train_stage} = "lora_sft" ]; then
        if [ ${model_item} = "llava-v1_6-vicuna-7b-lora_sft" ]; then
            train_config="../paddlemix/config/llava/v1_6/lora_sft_7b_argument_benchmark.json"
        fi
        if [ ${model_item} = "llava-v1_6-vicuna-13b-lora_sft" ]; then
            train_config="../paddlemix/config/llava/v1_6/lora_sft_13b_argument_benchmark.json"
        fi
        train_cmd="../paddlemix/tools/supervised_finetune.py ${train_config}"
    fi
        # train_cmd="../paddlemix/tools/supervised_finetune.py \
        #     --do_train true \
        #     --model_name_or_path liuhaotian/${model_path} \
        #     --dataset {\"train\":[{\"name\":\"chatml_dataset\",\"data_files\":\"./llava_bench_data/ScienceQA_val_500.json\",\"chat_template\":\"./llava_bench_data/chat_template.json\"}]} \
        #     --mixtoken false \
        #     --output_dir ./work_dirs/${model_item}_benchmark \
        #     --overwrite_output_dir true \
        #     --per_device_train_batch_size ${base_batch_size} \
        #     ${use_fp16_cmd} \
        #     --num_train_epochs ${max_epochs} \
        #     --recompute true \
        #     --gradient_accumulation_steps 1 \
        #     --sharding stage2 \
        #     --tensor_parallel_degree 1 \
        #     --sharding_parallel_degree 8 \
        #     --learning_rate 2e-04 \
        #     --mm_projector_lr 2e-5 \
        #     --weight_decay 0.0 \
        #     --warmup_ratio 0.03 \
        #     --lr_scheduler_type cosine \
        #     --logging_steps 1 \
        #     --save_steps 1000 \
        #     --evaluation_strategy no \
        #     --save_strategy no \
        #     --max_length 2048 \
        #     --do_eval false \
        #     --disable_tqdm true \
        #     --load_best_model_at_end false \
        #     --eval_with_do_generation false \
        #     --skip_memory_metrics false \
        #     --save_total_limit 1 \
        #     --lora true \
        #     --lora_rank 128 \
        #     --lora_alpha 256 \
        #     --lora_dropout 0.0 \
        #     --tensor_parallel_degree=1 \
        #     --sharding_parallel_degree=8 \
        #     --sharding="stage2" \
        #     --benchmark true
        #     "
    # fi
    if [ ${train_stage} = "pretrain" ]; then
        if [ ${model_item} = "llava-v1_6-vicuna-7b-pretrain" ]; then
            train_config="../paddlemix/config/llava/pretrain_7b_benchmark.json"
        fi
        if [ ${model_item} = "llava-v1_6-vicuna-13b-pretrain" ]; then
            train_config="../paddlemix/config/llava/pretrain_13b_benchmark.json"
        fi
        train_cmd="../paddlemix/examples/llava/pretrain.py ${train_config}"
    fi
        # train_cmd="../paddlemix/examples/llava/pretrain.py \
        #     --do_train true \
        #     --model_name_or_path paddlemix/llava/${model_path} \
        #     --dataset {\"train\":[{\"name\":\"chatml_dataset\",\"data_files\":\"./llava_bench_data/ScienceQA_val_500.json\"}]} \
        #     --output_dir ./work_dirs/${model_item}_benchmark \
        #     --overwrite_output_dir true \
        #     --per_device_train_batch_size ${base_batch_size} \
        #     ${use_fp16_cmd} \
        #     --num_train_epochs ${max_epochs} \
        #     --mixtoken false \
        #     --recompute true \
        #     --gradient_accumulation_steps 1 \
        #     --group_by_modality_length false \
        #     --learning_rate 1e-03 \
        #     --warmup_ratio 0.03 \
        #     --lr_scheduler_type cosine \
        #     --logging_steps 1 \
        #     --save_steps 1000 \
        #     --evaluation_strategy no \
        #     --save_strategy no \
        #     --max_length 2048 \
        #     --do_eval false \
        #     --disable_tqdm true \
        #     --save_total_limit 1 \
        #     --tensor_parallel_degree=1 \
        #     --sharding_parallel_degree=8 \
        #     --sharding="stage2" \
        #     --skip_memory_metrics false \
        #     --benchmark true
        #     "
    # fi


    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP) if [[ ${device_num} = "N1C1" ]];then
            echo "run ${run_mode} "
            train_cmd="python -u ${train_cmd}" 
        else
            rm -rf ./mylog   # 注意执行前删掉log目录
            train_cmd="python -u -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  ${train_cmd}" 
        fi
        ;;
    DP1-MP1-PP1)  echo "run run_mode: DP1-MP1-PP1" ;;
    *) echo "choose run_mode "; exit 1;
    esac
    
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 30m ${train_cmd} > ${log_file} 2>&1

    # if [ ${train_stage} = "sft" ]; then
    #     timeout 30m ${train_cmd} > ${log_file} 2>&1
    # fi
    # if [ ${train_stage} = "lora_sft" ]; then
    #     timeout 30m ${train_cmd} --lora_target_modules ["llama.layer.*q_proj.*","llama.layer.*k_proj.*","llama.layer.*v_proj.*","llama.layer.*gate_proj.*","llama.layer.*up_proj.*","llama.layer.*down_proj.*","llama.layer.*o_proj.*"] > ${log_file} 2>&1
    # fi
    # if [ ${train_stage} = "pretrain" ]; then
    #     timeout 30m ${train_cmd} --freeze_include '*llama*','*lm_head*' --freeze_exclude '*llama.mm_projector*' > ${log_file} 2>&1
    # fi

    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    # kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

# source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
_train       # 如果只产出训练log,不解析,可取消注释
# _run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开