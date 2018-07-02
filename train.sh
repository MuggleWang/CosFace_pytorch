#!/bin/bash
# Usage:
# ./train.sh GPU
#
# Example:
# ./train.sh 0,1,2,3

GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py 2>&1 | tee ./log/cosface_trainlog_`date +%Y%m%d%H%M`.log
