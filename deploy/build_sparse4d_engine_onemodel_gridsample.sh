#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# 对应 grid_sample 支路建图（无需 DFA 插件）

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dv3_1st_gridsample engine -> saving in ${ENV_SPARSEV41_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_SPARSEV41_ONNX} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_SPARSEV41_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_sparsev41_gridsample.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_sparsev41_gridsample.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_sparsev41_gridsample.json \
    --profilingVerbosity=detailed \
    >${ENVTRTDIR}/build_sparsev41_gridsample.log 2>&1

# STEP3: build frame > 2 sparse4dhead engine
echo "STEP3: build frame > 2 sparse4dv3_2nd_gridsample engine -> saving in ${ENV_SPARSEV42_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_SPARSEV42_ONNX} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_SPARSEV42_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_sparsev42_gridsample.json --exportProfile=${ENVTRTDIR}/buildProfile_sparsev42_gridsample.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_sparsev42_gridsample.json \
    --profilingVerbosity=detailed \
    >${ENVTRTDIR}/build_sparsev42_gridsample.log 2>&1
