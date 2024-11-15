#!/usr/bin/env bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
    model_item=${1:-"stable_diffusion_3-dreambooth_ft"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"1"}       # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16|bf16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递

    model_repo="PaddleMIX"          # (必选) 模型套件的名字
    speed_unit="sample/sec"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${6:-"20"}                 # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
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

    # 原生动态图
    export FLAG_FUSED_LINEAR=0
    export FLAGS_conv_workspace_size_limit=4096

    export FLAGS_cudnn_deterministic=True
    env |grep FLAG

    if [ ${fp_item} = "fp32" ]; then
        fp_item_cmd="no"
    else
        fp_item_cmd=${fp_item}
    fi
    echo "------------"
    ls;
    echo "------------"


    if [ ${model_item} = "stable_diffusion_3-dreambooth_ft" ];then
        train_cmd="
            ../ppdiffusers/examples/dreambooth/train_dreambooth_sd3.py \
            --pretrained_model_name_or_path=stable-diffusion-3-medium-diffusers-paddle-init  \
            --instance_data_dir=dog \
            --output_dir=trained-sd3 \
            --mixed_precision=${fp_item_cmd} \
            --instance_prompt=a-photo-of-sks-dog \
            --resolution=512 \
            --train_batch_size=${batch_size} \
            --gradient_accumulation_steps=4 \
            --learning_rate=5e-5 \
            --report_to=tensorboard \
            --lr_scheduler=constant \
            --lr_warmup_steps=0 \
            --max_train_steps=${max_iter} \
            --validation_prompt=A-photo-of-sks-dog-in-a-bucket \
            --validation_epochs=100 \
            --num_validation_images 1 \
            --seed=0 \
            --checkpointing_steps=10000
        "
    else
        export USE_PEFT_BACKEND=True
        train_cmd="
            ../ppdiffusers/examples/dreambooth/train_dreambooth_lora_sd3.py \
            --pretrained_model_name_or_path=stable-diffusion-3-medium-diffusers-paddle-init  \
            --instance_data_dir=dog \
            --output_dir=trained-sd3-lora \
            --mixed_precision=${fp_item_cmd} \
            --instance_prompt=a-photo-of-sks-dog \
            --resolution=512 \
            --train_batch_size=${batch_size} \
            --gradient_accumulation_steps=4 \
            --learning_rate=5e-5 \
            --report_to=tensorboard \
            --lr_scheduler=constant \
            --lr_warmup_steps=0 \
            --max_train_steps=${max_iter} \
            --validation_prompt=A-photo-of-sks-dog-in-a-bucket \
            --validation_epochs=100 \
            --num_validation_images 1 \
            --seed=0 \
            --checkpointing_steps=10000
        "
    fi 

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
    RUN_SLOW=${RUN_SLOW:-"true"}
    if [ "$RUN_SLOW" = "true" ]; then
        timeout 30m ${train_cmd} > ${log_file} 2>&1
    else
        echo "fast mode, only run 3m"
        timeout 3m ${train_cmd} > ${log_file} 2>&1
    fi
    # eval ${train_cmd}
    # eval "timeout 30m ${train_cmd} > ${log_file} 2>&1"
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
    echo ${train_cmd} >> ${log_file}
    cat ${log_file}
}

function _analysis_log(){
    # cd -
    analysis_log_cmd="python test_tipc/dygraph/dp/stable_diffusion_3/benchmark_common/analysis_log.py \
        ${model_item} ${log_file} ${speed_log_file} ${device_num} ${base_batch_size} ${fp_item}"
    echo ${analysis_log_cmd}
    eval ${analysis_log_cmd}
}

_set_params $@
str_tmp=$(echo `pip list|grep paddlepaddle-gpu|awk -F ' ' '{print $2}'`)
export frame_version=${str_tmp%%.post*}
export frame_commit=$(echo `python -c "import paddle;print(paddle.version.commit)"`)
export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
export model_commit=$(git log|head -n1|awk '{print $2}')
echo "---------frame_version is ${frame_version}"
echo "---------Paddle commit is ${frame_commit}"
echo "---------Model commit is ${model_commit}"
echo "---------model_branch is ${model_branch}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
