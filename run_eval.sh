#!/bin/bash

model_name="./models/Qwen2.5-0.5B"
log_dir="./logs"
log_file="qwen25_0.5b_eval.log"

tensor_parallel_size=1
data_parallel_size=1
gpu_memory_utilization=0.85

mkdir -p $log_dir

lm_eval --model vllm \
    --model_args pretrained=$model_name,tensor_parallel_size=$tensor_parallel_size,dtype=auto,gpu_memory_utilization=$gpu_memory_utilization,data_parallel_size=$data_parallel_size \
    --tasks piqa \
    --batch_size auto | tee $log_dir/$log_file
