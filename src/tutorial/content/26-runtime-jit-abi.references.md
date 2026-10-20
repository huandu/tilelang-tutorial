# 第 26 章 参考资料

版本以 [../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。
本章不使用任何「最新版」类措辞。

写作环境：macOS（Apple Silicon）、Python 3.11.7；**没有 NVIDIA GPU，没有昇腾 NPU**，
未安装 `tilelang` / `torch` / `torch_npu`。

因此本文件的核验状态只有三种含义：

* **已核验（2026-10-01）**：在锁定版本工作树上用 `sed -n '<N>p'` / `cat -n` /
  `grep -n` 定位并逐行阅读。**不等于运行过。**
* **已核验（2026-10-01，脚本）**：由本章自带的脚本在本机跑出结论，可复现。
* **部分核验**：只核验了「文件存在 / 命令无输出」这一层，正文对应结论标注为推断。

---

## 一、本章自带的核对工具（可复现全部行号声明）

| id | 位置 | 用途 | 状态 |
| --- | --- | --- | --- |
| T1 | `code/01-runtime-anchor-check/verify_runtime_anchors.py` | 两层核对：172 条「文件:行号」锚点表 + 全章 241 处行内引用的存在性/范围/非空扫描；同时覆盖 `_vendor/tilelang-v0.1.15/` 与 `_vendor/tilelang-ascend-v0.1.2.000/` | **已核验（2026-10-01，脚本）**：`citation scan: 241 line references checked`、`ok: 172/172 anchors verified`，退出码 0 |
| T2 | `code/02-cache-dir-inspect/inspect_cache_dir.py` | 复刻 `KernelCache` 的 manifest 校验与设备二进制缓存读取；`--selftest` 造 fixture 自检 | **已核验（2026-10-01，脚本）**：`--selftest` 三条判据全过，退出码 0；默认模式在本机退出码 2（无缓存目录） |
| T3 | `code/03-packed-abi-sim/packed_abi_sim.py` | 用纯 Python 复刻 packed ABI 的检查规则与报错文案 | **已核验（2026-10-01，脚本）**：`13/13 scenarios behaved as the source says`，退出码 0 |
| T4 | `code/04-torch-stream-call/torch_stream_call.py` | stream 语义实验（含反例注释） | **仅静态核对**：`py_compile` 通过；**未运行**（无 CUDA GPU） |
| T5 | `code/05-compile-only-inspect/inspect_via_compile_only.py` | 用官方 `compile_only` 工具在无卡机器上看 lower 产物 | **仅静态核对**：`py_compile` 通过；`compile_kernel_source` 签名核对 `tilelang/tools/compile_only.py:169`；**未运行**（未安装 tilelang） |
| T6 | `code/06-ascend-aclgraph/aclgraph_capture.py` | ACLGraph 捕获与回放最小示例 | **仅静态核对**：`py_compile` 通过；与 `_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py:21-59, :271-272` 逐行比对；**未运行**（无 NPU） |

图：`figures/fig-01-jit-lifecycle.md`、`figures/fig-02-cache-layers.md`、
`figures/fig-03-packed-abi.md`、`figures/fig-04-adapter-selection.md`。
四张图都是自绘的 Mermaid / ASCII 文本，没有从 corpus 或上游复制位图，
因此不需要 `figures/SOURCES.md`（说明见 `figures/README.md`）。

---

## 二、锁定的源码引用

> 行号由 T1 逐条核对。为节省篇幅，本节按主题分组，不再逐条重复「已核验（2026-10-01）」；
> 未特别注明者，核验状态一律为**已核验（2026-10-01）**，方式为阅读源码。

### 2.1 JIT、装饰与生命周期

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:94-103` | 模块级 `compile()` 入口 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:164-173` | `compile()` 转发给 `cached()`；说明 `cached()` 在这条链上收不到位置参数 |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:270-341` | `JITImpl` 的定义与 lazy/eager 两种模式的官方说明 |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:357-366` | `_kernel_cache` / `_call_form_cache` / `_tuner_cache` 三个字典 |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:379-392` | `get_tir` |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:394-415` | `_infer_jit_mode` 与 `initialize_jit_mode`（eager 不允许 `out_idx`） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:466-497` | `JITImpl.compile`（含 `debug_root_path` 落盘分支） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:499-505` | `parse_cache_key`（`_tuner_cache` 的键） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:519-564` | `JITImpl.__call__`：模式推断 → `parse_args` → 进程内缓存 → compile → eager 立即执行 |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:588-652` | `jit()` 装饰器与 `decorator()`：`prim_func(func, eager_jit=True)`、`inspect.getsource/signature` |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:49-91, :514-517` | `_CallFormCache` 快路径（仅 lazy 且无张量参数） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/language/eager/builder.py:1601-1730` | `JITFunc`：`_is_lazy_style`、`parse_args`、`get_tir`、`set_mode` |

### 2.2 JITKernel 与编译产物

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:41-64` | `JITKernel` 的类属性 |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:66-149` | 构造函数：解析 backend 上下文 → 编译 → `torch_function = adapter.func` |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:151-203` | `from_database`：从磁盘缓存重建 |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:206-222` | `__call__` 转发 |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:224-286` | `_compile_and_create_adapter`（含 tvm_ffi 的 ABI 打标） |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:288-322` | `_compile_artifact`：pass 上下文、instrumentation、`lower_with_context` |
| S19 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:295-296` | `enable_host_codegen` / `enable_device_compile` 来自 backend 声明 |
| S20 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:324-424` | `_create_adapter_from_artifact`：五个 adapter 的分派 |
| S21 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:426-495` | `_create_adapter_from_database` |
| S22 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:516-530` | `get_profiler` |
| S23 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532-552` | `get_kernel_source` / `get_host_source` |
| S24 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:554-628` | `run_once` / `show_source` / `export_sources` |
| S25 | `_vendor/tilelang-v0.1.15/tilelang/jit/abi.py:1-74` | 输出 ABI 的两个属性与 `prepare_tvm_ffi_callee_allocated_outputs` |
| S26 | `_vendor/tilelang-v0.1.15/tilelang/engine/param.py:12-68` | `KernelParam`（`from_buffer` / `from_var`） |
| S27 | `_vendor/tilelang-v0.1.15/tilelang/engine/param.py:153-187` | `CompiledArtifact`、`dump_kernel_params`、`load_kernel_params` |
| S28 | `_vendor/tilelang-v0.1.15/tilelang/jit/diagnostics.py:13-43` | `jit_phase` 与 `diagnostics_enabled` |

### 2.3 engine.lower

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S29 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:22-52` | host/device 判据（`calling_conv == DEVICE_KERNEL_LAUNCH`） |
| S30 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:55-62` | `extrac_params` |
| S31 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:65-89` | `host_codegen` |
| S32 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:92-106` | `device_codegen` 与 `device_codegen_without_compile` |
| S33 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:109-141` | `lower_to_host_device_ir`（`PreLowerSemanticCheck` → `context.lower` → `Filter`） |
| S34 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:144-213` | `lower_with_context` 与 `_lower_with_context_impl` |
| S35 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:216-244` | 独立入口 `lower()` |
| S36 | `_vendor/tilelang-v0.1.15/tilelang/engine/semantic_check.py:9-31` | `should_enable_prelower_semantic_check` 与 `PreLowerSemanticCheck`（四个检查器） |

### 2.4 缓存

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S37 | `_vendor/tilelang-v0.1.15/tilelang/cache/__init__.py:1-98` | 缓存派发表、`_resolve_cache_dispatch`、`cached()` |
| S38 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:35-58` | 缓存文件名与小节常量 |
| S39 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:60-83` | 平台相关的宿主编译参数（macOS 借 torch 的头文件路径） |
| S40 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:85-137` | `_get_tilelang_lib_stamp`：库内容哈希 |
| S41 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:139-150` | `_get_base_key` |
| S42 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:168-217` | 命名空间与三个根目录 |
| S43 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:220-241` | `_cleanup_stale_staging_dirs`（默认 3600 秒） |
| S44 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:243-284` | `_generate_key` |
| S45 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:286-453` | `cached()`：内存 → 磁盘 → 编译 → 回写 |
| S46 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:455-461` | `clear_cache` |
| S47 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:463-473` | `_get_cache_path` |
| S48 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:521-550` | 原子写工具（`_safe_write_file` / `_safe_write_executable`） |
| S49 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:552-639` | `_save_kernel_to_disk`：staging → fsync → 原子 rename |
| S50 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:641-731` | `_load_kernel_from_disk` |
| S51 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:733-747` | `_clear_disk_cache` |
| S52 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:770-848` | 必需文件、清单、完整性校验、rename 冲突判定 |
| S53 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:872-906` | `_build_kernel` → `JITKernel.from_database` |
| S54 | `_vendor/tilelang-v0.1.15/tilelang/cache/binary_cache.py:20-55` | `BinaryCache` 的设计说明（不可变条目、不删共享条目） |
| S55 | `_vendor/tilelang-v0.1.15/tilelang/cache/binary_cache.py:112-197` | `_finalize_key` / `get_path` / `load` / `save` |
| S56 | `_vendor/tilelang-v0.1.15/tilelang/cache/cuda_binary_cache.py:12-45` | CUDA 二进制缓存的 key 字段与「编译选项必须进 key」的注释 |
| S57 | `_vendor/tilelang-v0.1.15/tilelang/cache/ascend_binary_cache.py:12-42` | Ascend 二进制缓存（`.aibin`，多一个 `linker_options`） |
| S58 | `_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:59-120` | `tilelang_callback_cuda_compile`：先查二进制缓存再调 nvcc |
| S59 | `_vendor/tilelang-v0.1.15/tilelang/ascend/backend.py:15-49` | `tilelang_callback_ascend_compile`：bisheng + 二进制缓存 |
| S60 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:120-141` | codegen 末尾调用编译回调 |
| S61 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/kernel_cache.py:7-30` | tvm_ffi 的缓存子类（`executable.so`、`get_exportable_executable`） |
| S62 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/kernel_cache.py:7-21` | nvrtc 的缓存子类（额外 `kernel.py`） |
| S63 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/kernel_cache.py:10-76` | CuTeDSL 的缓存子类（`kernel.py` / `launcher.cpp` / `launcher_lib.so`） |
| S64 | `_vendor/tilelang-v0.1.15/tilelang/ascend/kernel_cache.py:7-20` | 昇腾的缓存子类（`device_kernel.asc` / `host_kernel.asc`） |

### 2.5 host ABI

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S65 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:38-42` | 两个属性名常量 |
| S66 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:234-302` | `AssumeRuntimeCheckExtractor` 与三个 `MakeAssert*` 工具 |
| S67 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:317-340` | `RequiresPackedAPI`：三类函数不改写 |
| S68 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:342-383` | `GetCalleeAllocatedOutputIndices` |
| S69 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:386-431` | `MakePackedAPI` 开头：`num_args` 的算法 |
| S70 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:433-482` | 新签名的四个形参与前两条断言 |
| S71 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:615-719` | 逐参数 type index 检查与绑定 |
| S72 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:721-783` | anchor 校验、`BindDLTensors` 调用、buffer 声明 |
| S73 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:1003-1070` | calling conv 改写、返回值改写、`params` 落地 |
| S74 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:1072-1122` | pass 工厂与注册名 `tl.MakePackedAPI` |
| S75 | `_vendor/tilelang-v0.1.15/src/transform/tvm_ffi_binder.h:28-186` | `TVMFFIABIBuilder` 的公开契约（含「错误经 FFI error slot 回传」的说明） |
| S76 | `_vendor/tilelang-v0.1.15/src/transform/tvm_ffi_binder.cc:30-76` | 约束文案与 `BinderAddAssert`（编译期可证伪就 `LOG(FATAL)`） |
| S77 | `_vendor/tilelang-v0.1.15/src/transform/tvm_ffi_binder.cc:294-328` | `RelaxedStrideCheck` |
| S78 | `_vendor/tilelang-v0.1.15/src/transform/tvm_ffi_binder.cc:330-424` | 非空检查与 shape 指针短路 |
| S79 | `_vendor/tilelang-v0.1.15/src/transform/tvm_ffi_binder.cc:426-620` | ndim / dtype 三元组 / shape 约束 / 亚字节总位数 |
| S80 | `_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:190-198` | `MakePackedAPI` 的 Python 包装 |
| S81 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:273` | CUDA pipeline 中的位置 |
| S82 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:184` | Ascend pipeline 中的位置 |

### 2.6 adapter 与框架集成

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S83 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/__init__.py:1-6` | 五个 adapter 的导出面 |
| S84 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/base.py:24-165` | `BaseKernelAdapter`：`_legalize_result_idx`、stream/device thunk、`CachedTextSource` |
| S85 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/tvm_ffi.py:1-45` | 设计说明（DLPack Exchange 取 work stream）与 NPU 交接 |
| S86 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/tvm_ffi.py:158-210` | `_uses_ffi_callee_allocated_output_abi` 与 `_process_dynamic_symbolic` |
| S87 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/tvm_ffi.py:212-346` | 默认路径的 torch 可调用体（入参校验、输出分配、DLPack 调用） |
| S88 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/tvm_ffi.py:348-384` | callee-allocated 路径 |
| S89 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/tvm_ffi.py:386-465` | `from_database` 与源码读取 |
| S90 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/adapter.py:31-50, :431-453` | stream/device provider 与显式传 stream |
| S91 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/adapter.py:99-176` | 构造：wrap → 编译 → `lib.init()` → `CythonKernelWrapper` |
| S92 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/cython_wrapper.pyx:86-164` | 张量校验（device / dtype / 静态 shape / stride / 连续性） |
| S93 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/cython_wrapper.pyx:172-236` | `forward`：stream 解析与输出分配 |
| S94 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/adapter.py:27-98` | NVRTC adapter 构造与 kernel 句柄 |
| S95 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/adapter.py:261-324` | `_wrap_forward_from_prebuild_lib` 与 stream 缺省 |
| S96 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/torch/metal.py:14-78` | `MetalKernelAdapter`（`torch.mps.compile_shader`） |
| S97 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/adapter.py:23-100` | `CuTeDSLKernelAdapter`（产物是 Python 模块） |
| S98 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:1073-1160` | `TLWrapper` 与 `TLPyWrapper` 的后端分派 |
| S99 | `_vendor/tilelang-v0.1.15/tilelang/ascend/torch_exchange.py:6-55` | NPU stream 交接与 raw stream 取值 |
| S100 | `_vendor/tilelang-v0.1.15/tilelang/ascend/torch_npu_exchange.pyx:41-60` | 打补丁 Torch 的 DLPack Exchange API 表 |
| S101 | `_vendor/tilelang-v0.1.15/tilelang/backend/execution_backend.py:18-32` | `ExecutionBackendSpec`（含 `supports_callee_allocated_outputs`） |
| S102 | `_vendor/tilelang-v0.1.15/tilelang/backend/module.py:150-231` | 可用后端枚举、`"auto"` 语义、`BackendContext` |
| S103 | `_vendor/tilelang-v0.1.15/tilelang/backend/module.py:310-335` | `create_backend_context` |
| S104 | `_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:24-37` | CUDA 的 `tvm_ffi → nvrtc → cython` 优先级；CuTeDSL 独立 |
| S105 | `_vendor/tilelang-v0.1.15/tilelang/ascend/execution_backend.py:7-21` | Ascend 的 `tvm_ffi → cython` 优先级 |
| S106 | `_vendor/tilelang-v0.1.15/tilelang/metal/execution_backend.py:9-12` | Metal 把 `torch` 排在第一位 |

### 2.7 环境变量、诊断与启动开销

| id | 引用位置 | 用途 |
| --- | --- | --- |
| S107 | `_vendor/tilelang-v0.1.15/tilelang/env.py:368, :371, :382` | `TILELANG_CACHE_DIR` / `TILELANG_PRINT_ON_COMPILATION` / `TILELANG_JIT_DIAGNOSTICS` |
| S108 | `_vendor/tilelang-v0.1.15/tilelang/env.py:435-463` | 缓存开关、lib stamp 开关、JIT 诊断开关 |
| S109 | `_vendor/tilelang-v0.1.15/tilelang/env.py:505-523` | lower trace 目录、默认 target / execution backend |
| S110 | `_vendor/tilelang-v0.1.15/tilelang/env.py:555-560` | 顶层 `enable_cache` / `disable_cache` / `is_cache_enabled` |
| S111 | `_vendor/tilelang-v0.1.15/tilelang/__init__.py:160-167, :200-221` | 顶层导出（`jit` / `compile` / `PassConfigKey` / 缓存开关） |
| S112 | `_vendor/tilelang-v0.1.15/tilelang/profiler/__init__.py:214-295` | `run_once` / `do_bench` |
| S113 | `_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:193-259` | 计时后端分派，含 `"cudagraph"` |
| S114 | `_vendor/tilelang-v0.1.15/tilelang/profiler/torch_bench.py:144-152` | `bench_with_cudagraph` 及其「minimal launch overhead」注释 |
| S115 | `_vendor/tilelang-v0.1.15/tilelang/tools/compile_only.py:1-42, :169-200` | 无卡编译工具与 `compile_kernel_source` |

### 2.8 版本边界（v0.1.15 里**没有**什么）

| id | 引用位置 / 命令 | 结论 | 核验状态 |
| --- | --- | --- | --- |
| S116 | `ls _vendor/tilelang-v0.1.15/tilelang/jit/adapter/` | 输出 `base.py cutedsl cython kernel_cache.py libgen.py nvrtc torch tvm_ffi.py utils.py wrapper.py`——**没有** `pto/` | 已核验（2026-10-01） |
| S117 | `grep -rn "pto" _vendor/tilelang-v0.1.15/tilelang/ascend/target.py` | 无输出；v0.1.15 没有 `target_is_pto` / `normalize_pto_target` | 已核验（2026-10-01） |
| S118 | `ls _vendor/tilelang-v0.1.15/examples/ \| grep -i aclgraph` 与 `grep -rli "aclgraph\|NPUGraph" _vendor/tilelang-v0.1.15/examples _vendor/tilelang-v0.1.15/tilelang` | 均无输出；v0.1.15 主干**没有** ACLGraph 示例（示例只在昇腾适配器仓库） | 已核验（2026-10-01） |
| S119 | `grep -rn "torch.autograd\|autograd.Function\|torch.library\|register_autograd" _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src` | 无输出；**没有** autograd 集成 | 已核验（2026-10-01） |
| S120 | `grep -rln "backward" _vendor/tilelang-v0.1.15/examples` | 一组成对的反向 kernel 示例（attention_sink、linear_attention、deepseek_mhc、ascend/flash_attention 等） | 已核验（2026-10-01） |
| S121 | `ls _vendor/tilelang-ascend-v0.1.2.000/tilelang/jit/adapter/` | 输出 `base.py ctypes cython dlpack.py libgen.py utils.py wrapper.py`——独立仓库的 adapter 家族与主干不同 | 已核验（2026-10-01） |
| S122 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/jit/__init__.py:36` | 独立仓库的 `execution_backend` 取值域是 `dlpack` / `ctypes` / `cython` | 已核验（2026-10-01） |
| S123 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/cache/__init__.py:70` | 独立仓库有顶层 `tilelang.cache.clear_cache()` | 已核验（2026-10-01） |

---

## 三、corpus 资料引用

抓取日期 2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3325.md` | https://github.com/tile-ai/tilelang/pull/3325 | `T.PersistentKernel` + `AutoPersistent`：动机、wave 折叠规则、性能对比表与测试条件（**未合入**） | 已核验（2026-10-01）：本地文件存在，逐段阅读；**合入状态以 S116–S118 的同类 grep 为准——本章未在 `_vendor/tilelang-v0.1.15/` 找到 `PersistentKernel`** |
| R2 | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/tilelang/jit/adapter/pto/` | https://github.com/tile-ai/tilelang | main 快照（commit `994b44e`）里的 PTO adapter 目录：`__init__.py` / `adapter.py` / `cython_wrapper.pyx` / `libgen.py` / `wrapper.py` | 已核验（2026-10-01）：目录与文件存在；**仅 main，非 v0.1.15** |
| R3 | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/tilelang/jit/adapter/pto/adapter.py:1-33` | 同上 | PTO adapter 的定位（"Compile and execute PTODSL kernels using the PTO launch wrapper"）与 stream provider | 已核验（2026-10-01）：阅读前 40 行；**仅 main** |
| R4 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20240715-1.md:27-49` | 华为昇腾开发者技术文章（GE 图引擎与 Host 调度） | Host Bound / Device Bound 的定义，以及 eager 单算子模式下发的五步 Host 流程 | 已核验（2026-10-01）：本地文件存在，阅读 §1–§2.2 |
| R5 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20260326-4.md:89-95` | 同上 | vLLM 启用 ACLGraph 后 rollout 场景端到端提升 15%–20%（带条件的框架级数字） | 已核验（2026-10-01）：本地文件存在；**该数字是上层框架的端到端结果，不是 TileLang kernel 的加速比** |
| R6 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/examples/aclgraph/README.md` | https://github.com/tile-ai/tilelang-ascend | ACLGraph 的三步流程、`capture model` 日志判据、图捕获的两条语义 | 已核验（2026-10-01）：与 `_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/README.md` 同源；本章引用锁定版本内的那份 |
| R7 | `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md:12-40` | （编者综合稿，非上游） | 与本章 2.1 节互为对照的「一次编译的全部参与者」图 | **部分核验**：编者二手材料。其第 37 行把 adapter 家族写成 `{cython,torch,nvrtc,pto,cutedsl}`，依据的是 corpus 的 **main 快照**；v0.1.15 实测**没有** `pto/`，见 S116。本章以 `_vendor/` 为准 |
| R8 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/runtime_internals/stubs.md:1-77` | https://tilelang.com/（runtime internals） | CUDA / ROCm / Ascend 三套 stub 的机制，解释「为什么在没有加速器的机器上也能 import」 | 已核验（2026-10-01）：本地为站点源文件；**未在正文展开**，仅作为 3.1 节「无卡也能 lower」的背景 |
| R9 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/compile_only.md:1-80` | https://tilelang.com/（tools/compile_only） | `compile_only` 的用法与两份示例输出（CPU C 与 CUDA C） | 已核验（2026-10-01）：本地为站点源文件；与 S115 的源码一致 |

---

## 四、上游链接

| 链接 | 用途 | 核验状态 |
| --- | --- | --- |
| https://github.com/tile-ai/tilelang | 主干仓库 | 已核验（2026-10-01）：本地 `_vendor/tilelang-v0.1.15/` 为 tag 检出 |
| https://github.com/tile-ai/tilelang/releases/tag/v0.1.15 | 版本 tag | **未核验（本机无网络访问该站点时未复核页面内容）**；版本号以 `_vendor/tilelang-v0.1.15/VERSION` 与 `git describe --tags` 为准 |
| https://github.com/tile-ai/tilelang/pull/3325 | `T.PersistentKernel` / `AutoPersistent` | 已核验（2026-10-01）：corpus 内有抓取正文 R1；**未合入状态由 `_vendor/` 侧检索确认** |
| https://github.com/tile-ai/tilelang-ascend | 昇腾适配器仓库 | 已核验（2026-10-01）：本地 `_vendor/tilelang-ascend-v0.1.2.000/` 为 tag 检出 |
| https://github.com/tile-ai/tilelang/issues/2115 | 多后端解耦 RFC | 已核验（2026-10-01）：corpus 内有抓取正文（`08-frontier-unmerged/raw/github-issue-2115.md`）；本章只在 2.4 节引用其结构性描述 |

---

## 五、口径与待补

### 5.1 已登记的口径差异

| 项 | 另一处的说法 | 本章实测 | 处理方式 |
| --- | --- | --- | --- |
| 缓存 key 里的 `args_repr` | 第 7 章 6.3 节把「编译参数（N、block_N、threads …）」列为 `args_repr` 的来源 | `cached()` 在 v0.1.15 只被 `compile()` 调用，而 `compile()` 的两个调用点（`tilelang/jit/__init__.py:466-477`、`tilelang/autotuner/param.py:67-77`）都只用关键字参数（`:164-173`），`*args` 恒为空，`args_repr` 恒为 `()`；编译期参数是**经由 TIR 文本**进 key 的 | 本章按源码写（4.4 节）；已在此登记，建议后续版本在下一轮统一第 7 章 6.3 节的表述。**不修改其他章节目录** |
| adapter 家族成员 | corpus 编者综合稿 `90-synthesis/02-compiler-internals-map.md:37` 与本章任务说明都列出 `pto` | v0.1.15 **没有** `pto/`（S116、S117）；main 快照才有（R2、R3） | 本章把 PTO 归入第 6 节「前沿」，并给出核验命令 |
| 「有没有 ACLGraph 支持」 | 容易读成「主干支持 / 不支持」 | ACLGraph 属于 CANN / `torch_npu`；v0.1.15 主干里只是**没有示例**（S118） | 正文 5.2 节显式区分「示例缺失」与「能力不可用」 |

### 5.2 尚需在真实硬件上验证的事项（本章不宣称）

1. `code/04-torch-stream-call/` 的 stream 竞态统计（无同步组的失败次数）；
   这是本章唯一「只有真机才能给出数字」的实验，正文与答案都刻意留空。
2. `code/05-compile-only-inspect/` 在装好 `tilelang==0.1.15` 的机器上的实际输出，
   尤其是 CPU 目标生成的 `*_kernel` 签名。
3. `code/06-ascend-aclgraph/` 在昇腾 950 上的 `capture model` 日志取值，
   以及 `graph.replay()` 后的数值一致性。
4. `MakePackedAPI` 生成的 host stub 在真实报错时的**确切文案**（本章引的是
   `ostringstream` 的拼接模板，与运行时实际输出可能有细节差异，
   例如 dtype 的打印格式）。

### 5.3 待补资料

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| TVM-FFI 的 DLPack Exchange API 正式规范 | 本章 4.11 节说明「stream 由运行时去问」时，只能引 TileLang 侧注释与 Cython 结构体定义；缺少一份权威的 API 语义说明，难以判断 `current_work_stream` 的契约细节 | `apache/tvm-ffi` 的 `include/tvm/ffi/extra/` 与 DLPack 规范文档 |
| v0.1.15 官方 release note | 本章用 `git describe --tags` 与源码检索确认版本边界，但没有 release note 来交叉核对「本版新增/移除」的官方表述 | https://github.com/tile-ai/tilelang/releases/tag/v0.1.15 |
| TileLang 官方对 host ABI 的开发者文档 | 本章的 ABI 描述完全来自 `src/transform/` 的实现与注释；若上游有一份 ABI 契约文档，可用来判断哪些字段属于稳定承诺 | `tile-ai/tilelang` 的 `docs/developer_guide/` 或 RFC 目录 |
| `cuLibraryLoadData` 路径下 NVRTC adapter 的 stream 归属说明 | 本章只确认了「缺省取 `torch.cuda.current_stream().cuda_stream`」这一行，未深究 cuda-python 层是否还做额外同步 | `cuda-python` 的 `cuda.bindings.driver` 文档与 `tilelang/jit/adapter/nvrtc/wrapper.py` |
