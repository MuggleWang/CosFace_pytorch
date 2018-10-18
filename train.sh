#!/bin/bash
# Usage:
# ./code/sphereface_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3

GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py 2>&1 | tee ./log/cosface_trainlog_`date +%Y%m%d%H%M`.log