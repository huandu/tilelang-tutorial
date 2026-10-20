# 第 24 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）；对照版本 `v0.1.14`
（commit `7e3bbf3703e769479757eb7ecec2f2626f3cd0a8`）。** 文中不出现「最新」这类不带版本号的
模糊表述；凡版本一律写明 `v0.1.15` 或 `v0.1.14`。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）；**没有 NVIDIA GPU，
没有昇腾 NPU，也没有安装 TileLang 或 CUDA toolkit**。因此：

* 本章所有代码示例的最高验证级别是「**仅静态核对**」（API 存在性与签名逐个在 `_vendor/`
  中比对过，`python3 -m py_compile` 通过），除 `code/01`–`code/03` 三个纯标准库模型属于
  「CPU 逻辑验证」（实际运行过）；
* **没有任何一条性能数字来自本机实测**；本章正文里没有给出任何「这样写更快」的断言。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

「已核验」的含义：在锁定版本的工作树上用 `grep -n` / `sed -n` 定位到该行区间，并阅读了
完整内容；行号区间与文件内容一致。**这不是运行验证。** 本章共核对 155 处源码引用。

### 后端注册、编译入口与 host/device 切分

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:123-143` | CUDA 后端的完整注册清单（pipeline / device codegen / execution backend / host codegen / callbacks） | 已核验（2026-10-01）：`sed -n '120,143p'`；文件共 143 行 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:59-121` | `tilelang_callback_cuda_compile`：gencode / fatbin / cubin 的选择与 nvcc 选项 | 已核验（2026-10-01）：`grep -n "def tilelang_callback_cuda_compile"` → 59 |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/cuda/codegen.py:8-19` | `is_cutedsl_target` / `is_plain_cuda_target` / 四个 build 回调 | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/backend/device_codegen.py:17-31` | `global_func_device_codegen`：把 TVM 全局函数包成 Python 回调 | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/backend/host_codegen.py:55-59` | `STANDARD_HOST_CODEGENS` = `c` / `llvm` | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:174-213` | `_lower_with_context_impl`：host 与 device 的分叉点 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:144-170` | `lower_with_context` docstring：`enable_host_codegen` / `enable_device_compile` 默认 False 的理由 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:26-51` | `has_device_kernel_launch` / `is_device_call_c_device` / `get_host_call` | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:138-139` | 用 `Filter` 把 lowering 结果切成 host_mod / device_mod | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:239-240` | `AnnotateDeviceRegions` → `SplitHostDevice` | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:284` | `CUDA_PIPELINE = PassPipeline("cuda", CUDAPassPipelineBody)`；文件共 284 行 | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:70` | `MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)`（v0.1.15 变化） | 已核验（2026-10-01） |

### device codegen（CUDA C）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S13 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:182-184` | `target.build.tilelang_cuda` / `..._without_compile` 的注册 | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:99-139` | `BuildTileLangCUDA` 四步：校验符号 → validate → AddFunction → 编译 | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:151-176` | `BuildTileLangCUDAWithoutCompile`：源码留在 `source_map["cuda"]`，用 dummy PTX 占位 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:74-82` | `cluster_dims` → `kClusterDimX/Y/Z` launch param tags | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:563-565` | `PrintFuncPrefix` 固定打印 `extern "C" __global__ ` | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:629-646` | `PrintExtraAttrs`：`__launch_bounds__(threads, min_blocks)` | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:596-627` | `ClusterInfoExtractor`：从 PrimFunc attrs 读 `cluster_dims` | 已核验（2026-10-01） |
| S20 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6560-6561` | `this->cluster_dims = ClusterInfoExtractor().extract(f)` | 已核验（2026-10-01） |
| S21 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6562-6580` | `AddFunction` 里 `__grid_constant__ const <type>` 参数打印 | 已核验（2026-10-01）：`grep -n "__grid_constant__"` → 6465 / 6574 |
| S22 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6542-6548` | PDL 时抑制 `__restrict__` 的判据 | 已核验（2026-10-01） |
| S23 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:986-994` | 64-lane 整型向量退化成 `longlong4`（v0.1.15 变化） | 已核验（2026-10-01） |
| S24 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6509-6628` | `AddFunction` 全体 | 已核验（2026-10-01）：`grep -n "void CodeGenTileLangCUDA::AddFunction"` → 6509；文件共 6631 行 |

### host wrapper 与 execution backend

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S25 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131` | `TMA_DESC_INIT_FUNC`：`cuTensorMapEncodeTiled(...)` 模板与 `globalStride + 1` | 已核验（2026-10-01） |
| S26 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:133-159` | `TMA_IM2COL_DESC_INIT_FUNC`：`cuTensorMapEncodeIm2col(...)` | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:161-200` | 普通启动与 cluster 启动两个模板（含 `NonPortableClusterSizeAllowed`） | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:68-85` | `PREDEF_INIT_FUNC` / `PREDEF_HOST_FUNC` | 已核验（2026-10-01） |
| S29 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:289-393` | `create_dispatch_func`：逐个 kernel 拼启动代码 | 已核验（2026-10-01） |
| S30 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:415-459` | `generate_tma_descriptor_args` | 已核验（2026-10-01） |
| S31 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:466-497` | `parse_source_information`：从 device/host 模块读 grid/block/smem/cluster | 已核验（2026-10-01） |
| S32 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:524-525` | 读 `tma_descriptor_args` 属性 | 已核验（2026-10-01） |
| S33 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:567-571` | `get_kernel_launch_code`：按 `cluster_dims` 选启动模板 | 已核验（2026-10-01） |
| S34 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:585-632` | `update_lib_code`：`source + include + init + host_func` | 已核验（2026-10-01） |
| S35 | `_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:24-35` | `CUDA_EXECUTION_BACKENDS`（tvm_ffi/nvrtc/cython）与 `CUTEDSL_EXECUTION_BACKENDS` | 已核验（2026-10-01）：文件共 37 行 |
| S36 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/libgen.py:41-80` | `LibraryGenerator.compile_lib`：cython 路径把 C 源码编成 `.so` | 已核验（2026-10-01） |
| S37 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/libgen.py:1-20`、`:37-50` | NVRTC 路径的动机与生命周期 | 已核验（2026-10-01） |
| S38 | `_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py:271-296`、`:309-366` | `get_ptx_from_source` / `get_sass_from_source` | 已核验（2026-10-01）；**未运行**（无 CUDA 工具链） |
| S39 | `_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py:634-650` | `have_tma`：`arch >= 9` | 已核验（2026-10-01） |

### `T.gemm` 的指令选择

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S40 | `_vendor/tilelang-v0.1.15/src/op/gemm.cc:33-48` | `ResolveGemmImpl`：按 target 找实现 | 已核验（2026-10-01） |
| S41 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375` | `Gemm::SelectInst` 的完整谓词链 | 已核验（2026-10-01） |
| S42 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:40-78` | `CheckWgmma`：dtype / 转置 / K 整除 | 已核验（2026-10-01） |
| S43 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:80-87` | `AllowTcgen5Mma`：sm_100 + 作用域 + meta 白名单 | 已核验（2026-10-01） |
| S44 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:89-97` | `AllowWgmma`：Hopper + M≥64 + warp 数 4 的倍数 | 已核验（2026-10-01） |
| S45 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:99-145` | `AllowVoltaMma` / `AllowTuringMma` | 已核验（2026-10-01） |
| S46 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:119-129` | `Use2CtaRequested` | 已核验（2026-10-01） |
| S47 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:157-181` | `FatalWgmmaUnavailable` / `FatalTcgen5Unavailable` | 已核验（2026-10-01） |
| S48 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:347-353` | 2CTA 无单 CTA 回退的注释 | 已核验（2026-10-01） |
| S49 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:180-216` | `ComputeDefaultWarpPartition`（含 policy 语义） | 已核验（2026-10-01） |
| S50 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:218-300` | `ComputeWgmmaWarpPartition`（`m_warp % 4 == 0`） | 已核验（2026-10-01） |
| S51 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:377-422` | `ComputeWarpPartition` / `ReuseExistingSharedLayout` / `RegisterCudaGemm` | 已核验（2026-10-01） |
| S52 | `_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h:22-177` | `GetTCGEN5MMAMeta`：atom_m/n/k 与 ws/2cta 的合法组合 | 已核验（2026-10-01）：`grep -n "GetTCGEN5MMAMeta"` → 23 |
| S53 | `_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/__init__.py:54-63` | 指令名 → python 实现的注册表 | 已核验（2026-10-01） |
| S54 | `_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_wgmma.py:22-40` | `infer_shared_layout`：按连续性选 swizzle 档位 | 已核验（2026-10-01） |
| S55 | `_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_tcgen05.py:34-60` | `GemmTCGEN5`：SS/TS 变体、`use_2cta`、`_warp_partition` | 已核验（2026-10-01） |
| S56 | `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py:35-60` | `TCGEN05DescriptorParams`（LBO/SBO/swizzle atom） | 已核验（2026-10-01） |
| S57 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:32-76` | CUDA 方言 `gemm`：`mbar` 与隐式等待的说明 | 已核验（2026-10-01） |
| S58 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:103-135` | `gemm_blockscaled`：SM100 vs SM120 两条路 | 已核验（2026-10-01）：`grep -n "Block-scaled GEMM for CUDA"` → 103 |
| S59 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:20-30` | `__all__` 含 `tcgen05_gemm` / `wgmma_gemm` 等（导出不一致，见 S116） | 已核验（2026-10-01） |
| S60 | `_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:117-122` | `TargetHasBulkCopy`（arch ≥ 90） | 已核验（2026-10-01） |
| S61 | `_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:104-116` | `TargetHasStmatrix` / `TargetHasTmem` | 已核验（2026-10-01） |
| S62 | `_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:85-90` | `TargetCudaHasAsyncCopy`（arch ≥ 80） | 已核验（2026-10-01） |

### `T.copy` / TMA

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S63 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-807` | `SelectCopyInstForLowering` 主判定链（含 `allow_load=false`） | 已核验（2026-10-01） |
| S64 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:810-891` | `ClassifyWarpSpecializedCopy`（producer 侧 load 允许 TMA） | 已核验（2026-10-01） |
| S65 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:143-261` | `CheckGlobalStrides` / `CheckBulkLoad` / `CheckBulkStore` | 已核验（2026-10-01） |
| S66 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:326-365` | `CheckBulkCopy1D` | 已核验（2026-10-01） |
| S67 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:407-425` | `CheckLDSMCopy`（:407-410）与 `CheckTMemLoad/Store`（:417-425） | 已核验（2026-10-01） |
| S68 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:531-570` | `MakeTmaUnavailableReason`（device-bound base、FP4 方向限制） | 已核验（2026-10-01） |
| S69 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:593-626` | `SelectTmaInst` / `SelectSyncLikeInst` | 已核验（2026-10-01） |
| S70 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:791-905` | `Copy::Lower` 分派与 `LowerCPAsync` | 已核验（2026-10-01） |
| S71 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:913-1040` | `LowerCluster`：bulk / 逐行 / SIMT 三条 SM-to-SM 路径 | 已核验（2026-10-01） |
| S72 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:1050-1120` | `ClusterCopyReplacer`（SIMT 回退） | 已核验（2026-10-01） |
| S73 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2108-2215` | `Copy::LowerBulk`：描述符构造、mbarrier 选择、multicast 参数 | 已核验（2026-10-01） |
| S74 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2061-2091` | `desc.global_stride` 用**字节**填充，并校验最内层 stride 为 1 | 已核验（2026-10-01） |
| S75 | `_vendor/tilelang-v0.1.15/src/cuda/op/tma_layout.h:40-46` | `kTmaMaxBoxDim = 256` / `kTmaSmemAddrAlign = 128` | 已核验（2026-10-01） |
| S76 | `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/copy_sm90.h:51-210` | `tma_store_cluster` 与 `tma_load_multicast` 系列 | 已核验（2026-10-01） |
| S77 | `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/copy_sm90.h:212-270` | 1D/2D `tma_load` | 已核验（2026-10-01） |
| S78 | `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/barrier.h:11-194` | mbarrier 原语与 `tma_store_arrive/wait` | 已核验（2026-10-01） |
| S79 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:288-299` | 给流水里的 copy 打 `no_implicit_async_commit_wait`（→ cp.async） | 已核验（2026-10-01） |
| S80 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:232-300` | `T.tma_copy` docstring：producer 半边与 store 的批量语义 | 已核验（2026-10-01） |
| S81 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/copy_op.py:19-70` | CUDA 方言 `copy` 的三个提示关键字 | 已核验（2026-10-01） |
| S82 | `_vendor/tilelang-v0.1.15/tilelang/language/builtin.py:370-380`、`:457-532` | `T.tma_store_wait` / `T.mbarrier_wait_parity` / `T.mbarrier_arrive` | 已核验（2026-10-01） |
| S83 | `_vendor/tilelang-v0.1.15/docs/programming_guides/cluster_tma.md` | TMA multicast / SM-to-SM 的官方说明与三路径表 | 已核验（2026-10-01）：随锁定版本的 docs 一起发布；**未在硬件上运行** |
| S84 | `_vendor/tilelang-v0.1.15/docs/programming_guides/instructions.md` | `T.copy` 各变体的语义合同（同步 / 显式异步 / 严格回退） | 已核验（2026-10-01） |

### warp specialization

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S85 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/warpgroup.py:19-59` | `T.ws` / `WarpSpecialize`：warp group = 128 线程 | 已核验（2026-10-01） |
| S86 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:27-38` | `allow_warp_specialized`：CUDA + `have_tma` + 未被禁用 | 已核验（2026-10-01） |
| S87 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:39-65` | `module_has_tma` / `_module_has_shared_barrier` | 已核验（2026-10-01） |
| S88 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:101-109` | 三个 WS pass 的调用顺序 | 已核验（2026-10-01） |
| S89 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:268-271` | `AnnotateWarpGroupRegAlloc` | 已核验（2026-10-01） |
| S90 | `_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:6-22`、`:35-48`、`:125-139`、`:143-156`、`:158-172` | `AutoWarpSpecialization` / `ProducerConsumerWarpSpecialized` / `InjectTcgen05Fence` / `AnnotateWarpGroupRegAlloc` / `MaterializeWSSchedule` 的 docstring | 已核验（2026-10-01）：文件共 196 行 |
| S91 | `_vendor/tilelang-v0.1.15/src/cuda/transform/ws_analysis.h:33-41` | `TileStmtKind` 分类枚举 | 已核验（2026-10-01） |
| S92 | `_vendor/tilelang-v0.1.15/src/cuda/transform/auto_schedule.cc:1-24` | 自动 WS 入口：opt-in、把语句打上 `tl.ws_op_id` | 已核验（2026-10-01） |
| S93 | `_vendor/tilelang-v0.1.15/tilelang/language/warp_specialize.py:33-66` | `WSSyncKind` / `WSRole`（含 `max_nreg`） | 已核验（2026-10-01） |
| S94 | `_vendor/tilelang-v0.1.15/tilelang/language/warp_specialize.py:69-80` | `WSPipeline`：full/empty mbarrier 对与多版本缓冲语义 | 已核验（2026-10-01） |
| S95 | `_vendor/tilelang-v0.1.15/tilelang/language/warp_specialize.py:107-172` | `WSSync` / `WSScope` / `WSSchedule` | 已核验（2026-10-01） |
| S96 | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:27`、`:151-165` | `T.WSID` 与 `T.annotate_ws_schedule` | 已核验（2026-10-01） |
| S97 | `_vendor/tilelang-v0.1.15/examples/warp_specialize/example_warp_specialize_gemm_copy_0_gemm_1.py` | 本章 3.3 节与 `code/04` 的原型 | 已核验（2026-10-01）：文件存在，代码逐行比对；**未运行** |
| S98 | `_vendor/tilelang-v0.1.15/examples/warp_specialize/example_warp_specialize_gemm_softpipe_stage2.py` | `num_stages=2` 与角色划分叠加的例子 | 已核验（2026-10-01）；**未运行** |
| S99 | `_vendor/tilelang-v0.1.15/src/cuda/transform/producer_consumer_ws.cc:483-500`、`:620-632` | producer/consumer 分类与 `kAsyncCopyNoImplicitCommitWait` 的注入 | 已核验（2026-10-01） |

### Blackwell / cluster / CuTe DSL

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S100 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:194-222` | `alloc_barrier`：arrive_count ≥ 1、`shared.barrier` scope | 已核验（2026-10-01） |
| S101 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:227-251` | `alloc_cluster_barrier` | 已核验（2026-10-01） |
| S102 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:254-300` | `alloc_tmem` docstring：TMEM 规格、分配规则、访问限制、自动释放 | 已核验（2026-10-01）：`grep -n "def alloc_tmem"` → 254 |
| S103 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:110-114` | `LowerBlackwell2SM` 必须在 `LayoutInference` 之前 | 已核验（2026-10-01） |
| S104 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:175-202` | `LowerSharedTmem` / mbarrier 前置检查 / `LowerSharedBarrier` / `FuseMBarrierArriveExpectTx` | 已核验（2026-10-01） |
| S105 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:256-266` | `InjectFenceProxy` → `ThreadSync` → `InjectTcgen05Fence` | 已核验（2026-10-01） |
| S106 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:12-40` | CUDA 方言 `Kernel`：`cluster_dims` 与「T.ClusterKernel 是必填版本」 | 已核验（2026-10-01） |
| S107 | `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:397-420` | `launch_kernel`：`cluster_dims` 归一化与写入 launch block | 已核验（2026-10-01） |
| S108 | `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/README.md` | TCGEN05 的「preview」定位、手工 mbarrier 协议、`InjectTcgen05Fence` 的边界 | 已核验（2026-10-01）；**未运行** |
| S109 | `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/gemm_tcgen5mma.py` | 本章 3.4 节与 `code/05` 的原型（注意其 `T.tcgen05_gemm` 调用，见 S116） | 已核验（2026-10-01）；**未运行** |
| S110 | `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py` | MXFP8 block-scaled 的 SM100 形态（C 在 TMEM） | 已核验（2026-10-01）：读取前 70 行；**未运行** |
| S111 | `_vendor/tilelang-v0.1.15/examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` | SM120 的 NVFP4 block-scaled 形态（C 在 fragment） | 已核验（2026-10-01）：文件存在；**未运行** |
| S112 | `_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:9-23` | CuTe DSL 后端复用 `CUDA_PIPELINE` | 已核验（2026-10-01）：文件共 23 行 |
| S113 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/checks.py:1-88` | 依赖门禁：`nvidia-cutlass-dsl >= 4.7.0`、禁止 4.3.4、要求 `cutlass.cute` | 已核验（2026-10-01） |
| S114 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/libgen.py:37-60` | CuTe DSL 库生成：写临时目录再 import | 已核验（2026-10-01） |
| S115 | `_vendor/tilelang-v0.1.15/tilelang/contrib/cutedsl/__init__.py`、`src/cuda/codegen/codegen_cutedsl.cc:210-270`、`src/cuda/codegen/rt_mod_cutedsl.cc:86` | 重导出 CUTLASS/CuTe、Python 打印器、codegen 注册 | 已核验（2026-10-01）：`grep -n "tilelang_cutedsl"` → `rt_mod_cutedsl.cc:86` |
| S116 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/__init__.py:88`、`:108-196`、`tilelang/language/__init__.py:11-12` | 显式 GEMM 变体未导出（4.3.6 节的结论） | 已核验（2026-10-01）：`grep -n "from .gemm_op import"` → 88；`grep -n "_CUDA_API_ALL = ("` → 108；facade 的 star-import 在 11 行。**未运行验证**（无 TileLang 环境） |

### 版本对照（`_vendor/tilelang-v0.1.14/` 与 `v0.1.15` 的差异）

核验方式：两个 tag 的工作树都用 `git clone --depth 1 --branch <tag>` 取得，**没有提交历史**，
因此差异全部来自文件级比较：

```bash
diff -rq _vendor/tilelang-v0.1.14/tilelang/cuda _vendor/tilelang-v0.1.15/tilelang/cuda
diff -rq _vendor/tilelang-v0.1.14/src/cuda      _vendor/tilelang-v0.1.15/src/cuda
diff -rq _vendor/tilelang-v0.1.14/tilelang/tileop _vendor/tilelang-v0.1.15/tilelang/tileop
diff -u  <v0.1.14 文件> <v0.1.15 文件>
```

| id | 差异 | 核验方式与状态 |
| --- | --- | --- |
| D1 | 新增 `src/cuda/op/gemm_blockscaled.cc`；`gemm.cc` 里的 blockscaled 分派与 `tl.get_tcgen5_blockscaled_instr_desc` 被移出 | 已核验（2026-10-01）：`diff -u .../src/cuda/op/gemm.cc ...` |
| D2 | 新增 `tilelang/cuda/op/gemm/gemm_tcgen05_blockscaled.py`、`tilelang/tileop/gemm_blockscaled/`；`gemm/__init__.py` 多注册 `cuda.tcgen05_blockscaled` | 已核验（2026-10-01） |
| D3 | `IsTmemBuffer()` 取代字面量 `"shared.tmem"`（`gemm.cc`、`copy_analysis.cc`） | 已核验（2026-10-01） |
| D4 | `CopyInstIsTMA` → `CopyInstIsTMALoad` / `CopyInstIsTMAStore`；`ClassifyWarpSpecializedProducerCopy` → `ClassifyWarpSpecializedCopy`（新增 shared→global 旁路） | 已核验（2026-10-01） |
| D5 | `tilelang/cuda/pipeline.py:70` 的 `MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)`；插入 `AutoWarpSpecialization()` | 已核验（2026-10-01） |
| D6 | 新增 pass config `tl.enable_auto_warp_specialization`；新增 `src/cuda/transform/auto_schedule.cc` 与 `auto_schedule/`；新增 `ws_analysis.{h,cc}` | 已核验（2026-10-01） |
| D7 | `src/cuda/transform/materialize_ws_schedule.cc` 大幅扩容（diff 3312 行） | 已核验（2026-10-01）：`diff -u ... | wc -l` |
| D8 | `codegen_cuda.{cc,h}`：新增 `EmitPackedX2Call` / `EmitPerLaneScalarCall` / `VisitStmt_(AssertStmtNode)`；`int8x64` → `longlong4`；fp4/fp8 逐 lane 打包构造 | 已核验（2026-10-01） |
| D9 | `src/cuda/target_utils.cc`：新增 `float4_e2m1fn → float8_e4m3` 可向量化 cast | 已核验（2026-10-01） |
| D10 | `tilelang/contrib/nvcc.py`、`tilelang/contrib/nvrtc.py` 两版**无差异** | 已核验（2026-10-01）：`diff -rq` 未列出 |
| D11 | `tilelang/jit/adapter/wrapper.py` 的差异几乎全在 `TLAscendSourceWrapper`（昇腾）；NV 侧只有 `uint64` 类型映射与 `get_stream_type` 返回类型放宽 | 已核验（2026-10-01） |
| D12 | examples：`diff` 显示 v0.1.15 新增 `ascend/` 与 `kpool/`（顶层条目 55 → 57） | 已核验（2026-10-01）：`ls | wc -l` 与 `diff <(ls A) <(ls B)` |

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），cuTile 一条为 2026-10-01，见 corpus 的
[README.md](../tilelang-corpus/README.md)。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | TMA 的单线程发射与异步 barrier 语义（第 295 行）、thread block cluster 与 DSM 的定义（第 249、263 行） | 已核验（2026-10-01）：本地为全文；行号已 `grep -n` 定位。**厂商公开资料，非实测** |
| R2 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md` | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | TMEM 的物理规格与分配规则（29726-29768）、`tcgen05.mma` 的 shape 表（29785-29810） | 已核验（2026-10-01）：本地为全文 Markdown（约 4 万行）；**厂商公开资料** |
| R3 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md` | https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html | 可移植 cluster 上限 8 / B200 非可移植 16（第 73 行）、`cudaOccupancyMaxActiveClusters` 建议（第 52 行）、DSM 的合并与 32 字节对齐建议（第 60-68 行）、SM 共享内存 228 KB（第 50 行） | 已核验（2026-10-01）：本地为全文；**厂商公开资料** |
| R4 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-compatibility-guide.md` | https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html | `sm_100a` / `compute_100a` 的 PTX 既不前向也不后向兼容（第 28 行）；`CUDA_FORCE_PTX_JIT=1` 的验证方法（第 39-46 行） | 已核验（2026-10-01）：本地为全文；**厂商公开资料** |
| R5 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608-fulltext.md` | https://arxiv.org/abs/2407.08608 | FlashAttention-3：warp specialization 的 producer/consumer 定义（第 173 行）、`setmaxnreg` 与寄存器再分配（第 176-178 行）、FP8 WGMMA 只支持 k-major（第 186-190 行） | 已核验（2026-10-01）：本地为全文；数字均为**论文自述**，本章未引用其性能数值 |
| R6 | `../tilelang-corpus/11-related-compilers/raw/github-nvidia-cutile-python.md` | https://github.com/NVIDIA/cutile-python | cuTile 走 Tile IR、需要 r580+ 驱动、`tileiras` 13.2 目前只支持 Blackwell 与 Ampere/Ada（第 64-70 行） | 已核验（2026-10-01）：本地为仓库 README 快照 |
| R7 | `../tilelang-corpus/11-related-compilers/raw/nvidia-cutedsl-docs.md` | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html | 仅用于确认「NVIDIA 官方有 CuTe DSL 这条路线」；本地文件只有 22 行的导航壳 | **部分核验**：文件存在但其内容只是文档站导航，没有正文。见「待补资料」第 1 条 |
| R8 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-c-programming-guide.md` | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html | 仅用于「CUDA 编程模型对 DSL 使用者也重要」这一句定位（文档目录结构说明） | **部分核验**：本地 48 行是文档的**总览页**，不是完整编程指南；对应的 `.html`（29 KB）也只是落地页。见「待补资料」第 2 条 |
| R9 | `../tilelang-corpus/README.md` | （corpus 自述，无独立上游链接） | corpus 的目录地图与抓取日期 | 已核验（2026-10-01） |
| R10 | `../tilelang-corpus/sources/sources.yaml` | （corpus 的源清单） | R1–R8 各条目的 id / 类别 / trust 等级 / 备注 | 已核验（2026-10-01）：`sed -n '345,560p'`、`:1150,1180p` |

---

## 上游链接核验说明

* 上表标注「已核验」的含义是：确认对应 corpus 文件在本地存在，且其头部记录了抓取日期与
  原始 URL；本章写作时**没有重新发起网络请求**（corpus 抓取日期为 2026-09-30，早于本章
  写作日期 2026-10-01，差异在 1 天内）。
* R7、R8 是本章最弱的两条资料：它们对应的 corpus 文件只有文档站的导航壳/总览页。
  因此本章**没有**基于这两条资料做任何技术论断——CUDA 编程模型的表述一律改引 R2
  （PTX ISA，本地全文）或 TileLang 自己的源码与 `docs/`。
* `_vendor/tilelang-v0.1.15/docs/` 是**随锁定版本发布的文档**，属于锁定版本的一部分，
  不是外部链接，因此在上表中归入源码引用（S83、S84）。

---

## 关于性能数字的说明

**本章正文中没有任何性能数字。** 原因有两层：

1. 写作环境没有 NVIDIA GPU，无法做任何实测；
2. 本章的主题是「代码生成与硬件特性」，回答的是「会选哪条指令、产物长什么样」，
   不是「哪条更快」。

正文中出现的与性能相关的表述全部是**定性且可溯源**的，例如「TMA 把地址生成交给硬件、
只需要一个线程发射」（R1 第 295 行）、「更大的 cluster 会降低可并行的 block 数」
（R3 第 73 行）。这些是厂商文档的陈述，不是本章的实测结论。

`../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md` 里出现的算力/带宽表格本章
没有引用；`arxiv-2407-08608-fulltext.md` 里的 740 TFLOPs / 1.2 PFLOPs 等数字也**未被引用**。

---

## 待补资料

1. **CuTe DSL 官方文档正文缺失。** corpus 的
   `11-related-compilers/raw/nvidia-cutedsl-docs.md` 只有 22 行，是文档站的导航壳；
   正文在 `cute_dsl_general/*.html` 下，未被抓取。建议后续版本补抓
   `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html`
   与同目录的 `dsl_code_generation.html`。本章因此只能对 CuTe DSL 做定位性描述，
   技术判断全部来自 TileLang 侧的 `checks.py` 与 `contrib/cutedsl/`。
2. **CUDA C++ Programming Guide 全文缺失。** corpus 里的
   `05-nvidia-backend/raw/nvidia-cuda-c-programming-guide.md`（48 行）与同名 `.html`
   （29 KB）都只是文档落地页，没有编程模型、`cp.async`、TMA、mbarrier 的章节正文。
   建议补抓 `https://docs.nvidia.com/cuda/cuda-c-programming-guide/` 的
   01-introduction / 02-basics / 04-special-topics 各章。本章涉及的 CUDA 语义因此改引
   PTX ISA 全文（R2）与 TileLang 的 `docs/programming_guides/*`（S83、S84）。
3. **`T.tcgen05_gemm` 导出不一致未做运行验证。** 4.3.6 节与 S116 的结论完全来自
   `__all__` / `import` / star-import 三处源码；因为没有可用的 TileLang 环境，
   无法用 `hasattr(T, "tcgen05_gemm")` 实际确认。建议后续版本在有 CUDA 的机器上跑一次
   一行验证，并把结论写进 `audit/`。
4. **`examples/` 未运行。** 本章引用了 6 个官方 example（S97、S98、S108–S111），
   全部只做了静态核对。它们的正确性、以及 `T.tcgen05_gemm` 那类调用能否真正跑通，
   都需要在 sm_90 / sm_100 机器上验证。
5. **NVIDIA Blackwell 架构白皮书未在 corpus。** 本章关于 Blackwell 的描述全部来自
   tuning guide 与 compatibility guide（R3、R4）加 PTX ISA（R2）。如果要写更细的
   SM 微架构（如每 SM 的 Tensor Core 数量、TMEM 带宽），需要补厂商白皮书。
