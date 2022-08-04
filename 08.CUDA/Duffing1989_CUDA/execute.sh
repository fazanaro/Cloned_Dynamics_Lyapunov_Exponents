#!/bin/bash
# $0 is the script name, $1 id the first ARG, $2 is second...
NAME="$1"
#
cp -rf /home/filipe/cuda-workspace/proj_CUDA_erro_Hindmarsh_Rose1984_ClDyn/${NAME}.cu /home/filipe/cuda-workspace/CUDA_HindmarshRose1984_erro/
#
nvcc ${NAME}.cu -Im -G -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -o ${NAME}
