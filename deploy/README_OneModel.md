# Deploy Sparse4D OneModel Pipeline In LocalWorkStation

> OneModel 模式将 backbone+head 合一，无需单独导出 backbone ONNX 和 engine。

## STEP1. Export Onnx (OneModel)
```bash
cd /path/to/SparseEnd2End
python deploy/export_OneModel_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt

# 可选参数：
#   --use_deformable_func false   # 非 NVIDIA 芯片走 grid_sample 支路
#   --o2                          # 仅导出 2nd frame ONNX
#   --save_onnx1 /path/to/1st.onnx
#   --save_onnx2 /path/to/2nd.onnx
```
onnx 将保存到 deploy/onnx 目录，输出如下:
>deploy/onnx
>├── export_head_onnx.log
>├── sparse4dv3_1st.onnx
>└── sparse4dv3_2nd.onnx

## STEP2. Compile Custom operator: deformableAttentionAggrPlugin.so
同标准流程，首先在 01_setEnv.sh 中设置环境变量，然后执行:
```bash
. deploy/dfa_plugin/tools/01_setEnv.sh
```
env 输出示例:
```bash
====================================================================================================================
||  Config Environment Below:
||  TensorRT LIB        : /mnt/env/tensorrt/TensorRT-8.5.1.7/lib
||  TensorRT INC        : /mnt/env/tensorrt/TensorRT-8.5.1.7/include
||  TensorRT BIN        : /mnt/env/tensorrt/TensorRT-8.5.1.7/bin
||  CUDA_LIB    : /usr/local/cuda-11.6/lib64
||  CUDA_INC    : /usr/local/cuda-11.6/include
||  CUDA_BIN    : /usr/local/cuda-11.6/bin
||  CUDNN_LIB   : /mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib
||  CUDASM      : sm_86
||  ENVBUILDDIR : build
||  ENVTARGETPLUGIN     : lib/deformableAttentionAggr.so
||  ENVONNX     : deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx
||  ENVEINGINENAME      : deploy/dfa_plugin/engine/deformableAttentionAggr.engine
||  ENVTRTDIR   : deploy/dfa_plugin/engine
====================================================================================================================
[INFO] Config Env Done, Please Check EnvPrintOut Above!
```
编译插件:
```bash
cd deploy/dfa_plugin
make -j8
```
make 输出示例:
```bash
1-Finish Compile CUDA Make Policy build/deformableAttentionAggr.cu.mk
2-Finish Compile CXX Make Policy build/deformableAttentionAggrPlugin.cpp.mk
make lib/deformableAttentionAggr.so
make[1]: Entering directory '/mnt/data/end2endlocal/tmp/SparseEnd2End/deploy/dfa_plugin'
3-Finish Compile CXX Objects : build/deformableAttentionAggrPlugin.cpp.o
4-Finish Compile CUDA Objects build/deformableAttentionAggr.cu.o
5-Finish Compile Target : lib/deformableAttentionAggr.so!
make[1]: Leaving directory '/mnt/data/end2endlocal/tmp/SparseEnd2End/deploy/dfa_plugin'
```

## STEP3. BUILD Sparse4D Engine (OneModel)
首先在 set_env_new.sh 中设置环境变量，然后执行:
```bash
cd -
. deploy/tools/set_env_onemodel.sh
bash deploy/build_sparse4d_engine_onemodel.sh
```
trt 产物如下:
>deploy/engine
>├── sparse4dv3_1st.engine
>├── sparse4dv3_2nd.engine
>├── build_sparsev41.log
>├── build_sparsev42.log
>├── buildLayerInfo_sparsev41.json
>├── buildLayerInfo_sparsev42.json
>├── buildOutput_sparsev41.json
>├── buildOutput_sparsev42.json
>├── buildProfile_sparsev41.json
>└── buildProfile_sparsev42.json
