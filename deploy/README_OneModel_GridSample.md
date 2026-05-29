# Deploy Sparse4D OneModel Pipeline (grid_sample) In LocalWorkStation

> OneModel + grid_sample 支路（`--use_deformable_func=False`）。
> 全程走原生 PyTorch grid_sample 采样，无需编译 DFA 自定义 CUDA 插件，适合非 NVIDIA 芯片或快速验证。

## STEP1. Export Onnx (OneModel, grid_sample)
```bash
cd /path/to/SparseEnd2End
python deploy/export_OneModel_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt --use_deformable_func=False

# 可选参数：
#   --o2                          # 仅导出 2nd frame ONNX
#   --save_onnx1 /path/to/1st.onnx
#   --save_onnx2 /path/to/2nd.onnx
```
onnx 将保存到 deploy/onnx 目录，输出如下:
>deploy/onnx
>├── export_head_onnx.log
>├── sparse4dv3_1st_gridsample.onnx
>└── sparse4dv3_2nd_gridsample.onnx

## STEP2. 无需编译 DFA 插件
grid_sample 支路使用 PyTorch 原生算子建图，不依赖 `DeformableAttentionAggr` CUDA 自定义算子，因此**跳过**标准流程中的插件编译步骤。

## STEP3. BUILD Sparse4D Engine (OneModel, grid_sample)
首先在 set_env_onemodel_gridsample.sh 中设置环境变量，然后执行:
```bash
. deploy/tools/set_env_onemodel_gridsample.sh
bash deploy/build_sparse4d_engine_onemodel_gridsample.sh
```
trt 产物如下:
>deploy/engine
>├── sparse4dv3_1st_gridsample.engine
>├── sparse4dv3_2nd_gridsample.engine
>├── build_sparsev41_gridsample.log
>├── build_sparsev42_gridsample.log
>├── buildLayerInfo_sparsev41_gridsample.json
>├── buildLayerInfo_sparsev42_gridsample.json
>├── buildOutput_sparsev41_gridsample.json
>├── buildOutput_sparsev42_gridsample.json
>├── buildProfile_sparsev41_gridsample.json
>└── buildProfile_sparsev42_gridsample.json
