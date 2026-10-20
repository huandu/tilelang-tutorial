# 图 6：v0.1.14 → v0.1.15 在 NV codegen 上的变化地图

用 `diff -rq _vendor/tilelang-v0.1.14/<dir> _vendor/tilelang-v0.1.15/<dir>` 得到。

```text
src/cuda/op/
├─ gemm.cc                  blockscaled 分派被移出（-）
├─ gemm_blockscaled.cc      【新增】blockscaled 专用分派
├─ copy_analysis.cc         CopyInstIsTMA 拆分；ClassifyWarpSpecializedCopy 泛化
└─ target_utils.cc          fp4 -> fp8 可向量化 cast（+）

src/cuda/transform/
├─ auto_schedule.cc         【新增】自动 warp specialization 入口
├─ auto_schedule/           【新增】role_based_scheduler / preprocess_ir / memory_detector
├─ ws_analysis.{h,cc}       【新增】TileStmtKind 分类（从 producer_consumer_ws.cc 抽出）
└─ materialize_ws_schedule.cc  大幅扩容（diff 3312 行）

src/cuda/codegen/
├─ codegen_cuda.{cc,h}      EmitPackedX2Call / EmitPerLaneScalarCall / AssertStmt；
│                           int8x64 改为 longlong4；fp4/fp8 逐 lane 打包构造
└─ codegen_cutedsl.cc       有改动（CuTe DSL 打印器）

tilelang/
├─ cuda/pipeline.py         MaterializeKernelLaunch(lower_thread_binding=True,
│                           default_threads=128)；插入 AutoWarpSpecialization
├─ cuda/op/gemm/            gemm_tcgen05_blockscaled.py【新增】；__init__ 多注册一条
├─ cuda/language/           CUDA 方言显式 shadow 常用构造（T.Kernel / T.copy /
│                           T.gemm / T.gemm_blockscaled / T.atomic_add / T.Parallel ...）
├─ tileop/gemm_blockscaled/ 【新增】包
├─ transform/pass_config.py 新增 tl.enable_auto_warp_specialization
└─ jit/adapter/wrapper.py   仅 Ascend 相关变化 + uint64 类型映射

examples/
├─ ascend/                  【新增】
├─ kpool/                   【新增】
└─ gemm_tcgen05/README.md, blockscaled_gemm_sm100/mxfp8_illustrated.md 等有改动
```

无变化的 NV 相关文件（`diff -rq` 未列出）：`tilelang/contrib/nvcc.py`、
`tilelang/contrib/nvrtc.py`、`tilelang/jit/adapter/nvrtc/*`（除 wrapper.py 外）。

注意：两个 vendor 树都是 `git clone --depth 1 --branch <tag>`，没有提交历史，
所以上面的结论全部来自**文件级 diff**，不是`git log`。
