#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# 对应 export_OneModel_onnx.py --use_deformable_func=False 导出的 ONNX（grid_sample 支路，无需 DFA 插件）

################***EnvVersion-1***#################
###     LinuxPlatform:            x86_64                                                        ###
###     TensorRT :                    8.5.1.7                                                          ###
###     CUDA :                             11.6                                                            ###
###     cuDNN:                          8.6.0.163                                                    ###
###     CUDA capability :        sm_86                                                         ###
################***EnvVersion-1***#################

################***EnvVersion-2***#################
###     LinuxPlatform:            x86_64                                                        ###
###     TensorRT :                    8.6.1.6                                                          ###
###     CUDA :                             11.6                                                            ###
###     cuDNN:                          8.6.0.163                                                    ###
###     CUDA capability :        sm_86                                                         ###
################***EnvVersion-2***#################

EnvVersion=2
if [ $EnvVersion = 1 ]; then
    export ENV_TensorRT_LIB=/mnt/env/tensorrt/TensorRT-8.5.1.7/lib
    export ENV_TensorRT_INC=/mnt/env/tensorrt/TensorRT-8.5.1.7/include
    export ENV_TensorRT_BIN=/mnt/env/tensorrt/TensorRT-8.5.1.7/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.6/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.6/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.6/bin
else
    export ENV_TensorRT_LIB=/usr/local/TensorRT-8.6.1.6/lib
    export ENV_TensorRT_INC=/usr/local/TensorRT-8.6.1.6/include
    export ENV_TensorRT_BIN=/usr/local/TensorRT-8.6.1.6/bin
    export ENV_CUDA_LIB=/usr/local/cuda/lib64
    export ENV_CUDA_INC=/usr/local/cuda/include
    export ENV_CUDA_BIN=/usr/local/cuda/bin
fi
export ENV_cuDNN_LIB=/home/adt/Software/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib

if [ ! -f "${ENV_TensorRT_BIN}/trtexec" ]; then
    echo "[ERROR] Failed to Find ${ENV_TensorRT_BIN}/trtexec!"
    return
fi

# Part1
export PATH=$ENV_TensorRT_BIN:$CUDA_BIN:$PATH
export LD_LIBRARY_PATH=$ENV_TensorRT_LIB:$ENV_CUDA_LIB:$ENV_cuDNN_LIB:$LD_LIBRARY_PATH

# Part2 Build TensorRT engine (grid_sample 支路无需 DFA 插件).
export ENVTRTDIR=deploy/engine

export ENV_SPARSEV41_ONNX=deploy/onnx/sparse4dv3_1st_gridsample.onnx
export ENV_SPARSEV41_ENGINE=${ENVTRTDIR}/sparse4dv3_1st_gridsample.engine

export ENV_SPARSEV42_ONNX=deploy/onnx/sparse4dv3_2nd_gridsample.onnx
export ENV_SPARSEV42_ENGINE=${ENVTRTDIR}/sparse4dv3_2nd_gridsample.engine

echo "===================================================================================================================="
echo "||  Config Environment Below:"
echo "||  TensorRT LIB \t: $ENV_TensorRT_LIB"
echo "||  TensorRT INC \t: $ENV_TensorRT_INC"
echo "||  TensorRT BIN \t: $ENV_TensorRT_BIN"
echo "||  CUDA_LIB \t: $ENV_CUDA_LIB"
echo "||  CUDA_INC \t: $ENV_CUDA_INC"
echo "||  CUDA_BIN \t: $ENV_CUDA_BIN"
echo "||  CUDNN_LIB \t: $ENV_cuDNN_LIB"
echo "||  ENVTRTDIR\t: ${ENVTRTDIR} (deploy/engine)"
echo "||  "
echo "||  NOTE: grid_sample 支路无需 DFA 插件，trtexec 构建时不要传 --plugins"
echo "||  ENV_SPARSEV41_ONNX\t: $ENV_SPARSEV41_ONNX"
echo "||  ENV_SPARSEV41_ENGINE\t: $ENV_SPARSEV41_ENGINE"
echo "||  ENV_SPARSEV42_ONNX\t: $ENV_SPARSEV42_ONNX"
echo "||  ENV_SPARSEV42_ENGINE\t: $ENV_SPARSEV42_ENGINE"
echo "===================================================================================================================="
echo "[INFO] Config Env Done, Please Check EnvPrintOut Above!"
