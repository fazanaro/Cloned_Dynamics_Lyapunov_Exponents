#!/bin/bash
# $0 is the script name, $1 id the first ARG, $2 is second...
NAME="$1"
nvcc ${NAME}.cu -Im -G -O0 -keep -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -o ${NAME}
#nvcc ${NAME}.cu -Im -G -O0 -keep --ptxas-options="-v" -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -o ${NAME}
#nvcc ${NAME}.cu -gencode arch=compute_20,code=sm_20 -o ${NAME}
