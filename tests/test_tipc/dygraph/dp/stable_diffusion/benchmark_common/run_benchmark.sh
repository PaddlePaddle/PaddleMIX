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
    model_item=${1:-"stable_diffusion-098b_pretrain"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"2"}       # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16|bf16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
 
    model_repo="PaddleMIX"          # (必选) 模型套件的名字
    speed_unit="sample/sec"         # (必选)速度指标单位
    skip_steps=4                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${6:-"500"}                 # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${7:-"5"}                # (可选)
    is_large_model=False           # (可选)普通模型默认为False，如果添加大模型且只取一条ips设置为True

    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
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

    # add some flags
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_conv_workspace_size_limit=4096

    if [ ${model_item} = "stable_diffusion-098b_lora" ];then
        if [ ${fp_item} = "fp16O1" ]; then
            use_fp16_cmd="--mixed_precision fp16 --fp16_opt_level O1"
        fi
        if [ ${fp_item} = "bf16O1" ]; then
            use_fp16_cmd="--mixed_precision bf16 --fp16_opt_level O1"
        fi

        FUSED=False
        if [ ${fp_item} = "fp16O2" ]; then
            use_fp16_cmd="--mixed_precision fp16 --fp16_opt_level O2"
            # FUSED=True
            # [operator < fused_gemm_epilogue_grad > error]

        fi
        if [ ${fp_item} = "bf16O2" ]; then
            use_fp16_cmd="--mixed_precision bf16 --fp16_opt_level O2"
            # FUSED=True
            # [operator < fused_gemm_epilogue_grad > error]
        fi
        rm -rf ./outputs

        # use fused linear in amp o2 level
        export FLAG_FUSED_LINEAR=${FUSED}
        train_cmd="../ppdiffusers/examples/text_to_image/train_text_to_image_lora_benchmark.py \
                --output_dir ./outputs \
                --train_batch_size=${batch_size} \
                --gradient_accumulation_steps=1 \
                --logging_steps=5 \
                --max_train_steps=${max_iter} \
                --pretrained_model_name_or_path=./CompVis-stable-diffusion-v1-4-paddle-init \
                --resolution=512 --random_flip \
                --checkpointing_steps=99999 \
                --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
                --seed=42 \
                --gradient_checkpointing \
                --dataloader_num_workers 8 \
                --enable_xformers_memory_efficient_attention \
                ${use_fp16_cmd}
                "
    else
        if [ ${fp_item} = "fp16O1" ]; then
            use_fp16_cmd="--fp16 True --fp16_opt_level O1"
        fi
        if [ ${fp_item} = "bf16O1" ]; then
            use_fp16_cmd="--bf16 True --fp16_opt_level O1"
        fi

        FUSED=False
        if [ ${fp_item} = "fp16O2" ]; then
            use_fp16_cmd="--fp16 True --fp16_opt_level O2"
            FUSED=True
        fi
        if [ ${fp_item} = "bf16O2" ]; then
            use_fp16_cmd="--bf16 True --fp16_opt_level O2"
            FUSED=True
        fi
        rm -rf ./outputs

        export FLAG_USE_EMA=0
        export FLAG_BENCHMARK=1
        export FLAG_RECOMPUTE=1
        export FLAG_XFORMERS=1
        # use fused linear in amp o2 level
        export FLAG_FUSED_LINEAR=${FUSED}

        train_cmd="../ppdiffusers/examples/stable_diffusion/train_txt2img_laion400m_trainer.py \
                --do_train \
                --output_dir ./outputs \
                --per_device_train_batch_size ${batch_size} \
                --gradient_accumulation_steps 1 \
                --learning_rate 1e-4 \
                --weight_decay 0.01 \
                --max_steps ${max_iter} \
                --lr_scheduler_type "constant" \
                --warmup_steps 0 \
                --image_logging_steps 1000 \
                --logging_steps 5 \
                --resolution 256 \
                --save_steps 10000 \
                --save_total_limit 20 \
                --seed 23 \
                --dataloader_num_workers 8 \
                --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
                --file_list ./data/filelist/train.filelist.list \
                --model_max_length 77 \
                --max_grad_norm -1 \
                --disable_tqdm True \
                --overwrite_output_dir \
                ${use_fp16_cmd}
            "
    fi

    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP-recompute) if [[ ${device_num} = "N1C1" ]];then
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

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
# _train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开