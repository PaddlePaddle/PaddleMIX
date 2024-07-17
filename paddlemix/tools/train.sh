#!/bin/bash

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}
#
echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done



unset GLOG_vmodule GLOG_v


export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=4096


# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8 
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3

# 启动方式
cuda_version=`nvidia-smi |grep "CUDA Version" |awk '{print $9}' |awk -F'.' '{print $1}'`
if [ ${cuda_version} != "12" ];then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
fi

##多机
##master=`cat /etc/host | head -n 1 | awk '{print $1}'`

#单机
master=$master_train

port=36677

export CUDA_VISIBLE_DEVICES=2,3

python3.9 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    ${script:-paddlemix/tools/supervised_finetune.py}  \
    $@



