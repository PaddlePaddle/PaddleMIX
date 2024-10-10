#!/bin/bash

#PROCESS="trtexec"
PROCESS="2053"
#echo $PROCESS

#LOG="./memlog.txt"
LOG= "memlog.txt"

#echo "$LOG"
#删除上次的监控文件
if [ -f "$LOG" ];then
    rm "$LOG"
fi

#过滤出需要的进程ID
#PID=$(ps aux| grep $PROCESS | grep -v 'grep' | awk '{print $2;}')
PID=$PROCESS
echo "process is $PID"

while [ "$PID" != "" ]
do
    # cpu_mem_kb=$(cat /proc/$PID/status | grep RSS | cut -d' ' -f 2)

    # if [ -z "$cpu_mem_kb" ]; then
    #     cpu_mem_kb=$(cat /proc/$PID/status | grep RSS | cut -d' ' -f 3)
    # fi

    #cpu_util=$(top -n 1 -d 0.01 -p $PID  | grep "$PROCESS" | awk '{print $10}')
    gpu_info=$(nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)

    #cat /proc/$PID/status | grep RSS | cut -d' ' -f 2  >> "$LOG"
    #nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$LOG".gpu

    #echo $cpu_mem_kb, $cpu_util, $gpu_info >> "$LOG"

    # cpu_mem_mb=$((${cpu_mem_kb:-0}/1024))
    # echo $cpu_mem_mb, $gpu_info >> "$LOG"
    sleep 0.1
    PID=$(ps -eo pid | grep $PROCESS | grep -v 'grep')
    PID=$(echo $PID | sed 's/^[ \t]*//g')
done