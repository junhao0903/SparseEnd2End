#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dv3_1st engine -> saving in ${ENV_SPARSEV41_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_SPARSEV41_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_SPARSEV41_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_sparsev41.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_sparsev41.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_sparsev41.json \
    --profilingVerbosity=detailed \
    >${ENVTRTDIR}/build_sparsev41.log 2>&1

# STEP3: build frame > 2 sparse4dhead engine
echo "STEP3: build frame > 2 sparse4dv3_2nd engine -> saving in ${ENV_SPARSEV42_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_SPARSEV42_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_SPARSEV42_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_sparsev42.json --exportProfile=${ENVTRTDIR}/buildProfile_sparsev42.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_sparsev42.json \
    --profilingVerbosity=detailed \
    >${ENVTRTDIR}/build_sparsev42.log 2>&1
