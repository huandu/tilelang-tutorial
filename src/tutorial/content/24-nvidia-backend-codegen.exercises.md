# 第 24 章 习题

说明：

* 难度：`★` 基础 / `★★` 需要综合本章内容 / `★★★` 动手或推理链较长。
* “需要硬件”：`否` 表示一台能跑 Python 的机器即可；`是（GPU）` 表示需要真实的
  NVIDIA GPU。**本章没有任何一题强制要求硬件**，需要 GPU 的题都标了“可选”。
* 所有题目只依赖第 24 章及更早章节的知识；引用源码时一律指向
  `_vendor/tilelang-v0.1.15/`。

---

## E1 概念题 ★ 不需要硬件

一次 `tilelang.compile(...)` 之后，NVIDIA 侧会得到哪几种产物？请分别说明：

1. device 侧的 CUDA C 源码是在哪个函数、哪个文件里生成的？
2. 默认情况下，device 源码会被编译成 cubin 吗？为什么？
3. device 的 cubin / PTX 与 host 的 `call()` 是在什么时候被拼到一起的？

---

## E2 概念题 ★ 不需要硬件

`tilelang/cuda/execution_backend.py:24-31` 注册了三种 execution backend。请回答：

1. 三种分别是什么？各自的默认可用性与依赖是什么？
2. 为什么说“读 `get_kernel_source()` 得到的结论对三种 backend 都成立”？
3. 如果你的机器上只装了 GPU 驱动、没有完整 CUDA toolkit，哪一种 backend 仍然可用？
   依据是什么？

---

## E3 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375`，回答：

1. 为什么 `op.isWgmma_` 这一支在约束不满足时是 `LOG(FATAL)` 而不是返回 `kCudaMMA`？
2. `Use2CtaRequested(op)` 为什么不设“回退到单 CTA”的分支？请在源码注释里找到理由。
3. `TargetIsVolta(target) && !AllowVoltaMma(op)` 这一支返回 `kCudaFMA`；那
   `TargetIsVolta(target) && AllowVoltaMma(op)` 会走到哪里？返回值是什么？

---

## E4 概念题 ★★ 不需要硬件

“默认的 `T.copy` 只有 store 会自动降成 TMA”这句话：

1. 在 `SelectCopyInstForLowering` 的哪一行代码里体现出来？请写出那一行。
2. 一个 `T.copy(A_global, A_shared)` 想走 TMA，有哪三种写法/场景？分别对应哪一段代码？
3. 一句**裸写、不在 `T.Pipelined` 里**的 `T.copy(A_global, A_shared)` 在 sm_90 上最终会被
   降成什么？为什么不是 `cp.async`？

---

## E5 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:143-261`，判断下面五个 copy 能否
走描述符型 TMA（`kBulkLoad` / `kBulkStore`），并写出被拒绝的那一条具体违反了哪条检查：

1. sm_90，`global → shared.dyn`，dtype=f16，最后一维 64 个元素，row-major；
2. sm_90，`global → shared.dyn`，dtype=f16，最后一维 30 个元素；
3. sm_90，`global → shared.dyn`，dtype=f16，shape=(4096,4096) 但最内层 stride = 2；
4. sm_90，`global → shared.dyn`，dtype=f16，stride 为 (12, 1)（单位：元素）；
5. sm_80，`shared.dyn → global`，dtype=f16，其余条件与 1 相同。

---

## E6 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:40-97`，回答：

1. `AllowWgmma` 要求哪三个条件？（提示：一个是架构，一个是 M，一个是 warp 数）
2. 同样是 fp8×fp8 → f32，`trans_A=True, trans_B=True` 与 `trans_A=False, trans_B=True`
   哪个能被 WGMMA 接受？为什么？
3. f16×f16 → f32 对 `trans_A` / `trans_B` 有要求吗？为什么这两类 dtype 的规则不一样？

---

## E7 概念题 ★ 不需要硬件

`GemmWarpPolicy` 的 `FullRow` / `FullCol` / `Square` 三种取值，分别影响输出里的什么？
请在 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc` 里指出：

1. 哪一个函数用到了 policy；
2. TCGEN05 与 FMA 两种指令为什么不看 policy（提示：看 `ComputeWarpPartition` 的前两个分支）。

---

## E8 动手题 ★★ 不需要硬件

用 `code/01-copy-inst-selection/copy_inst_selection.py`：

1. 先运行一次，记录全部判定结果；
2. 把其中一个场景的 `arch` 从 `sm_90` 改成 `sm_100`，观察结果是否变化，并解释原因
   （提示：`target_has_tmem`）；
3. 构造一个能触发 `cluster_mask requires descriptor-based TMA` 报错的场景，并说明是哪条
   检查拒绝了它。

---

## E9 动手题 ★★ 不需要硬件

用 `code/02-gemm-inst-selection/gemm_inst_selection.py`：

1. 找出一组 `(M, N, K, dtype, scope)` 使得在 sm_100 上选 `cuda.tcgen05`，在 sm_90 上选
   `cuda.wgmma`，并说明原因；
2. 找出一组使得在 sm_100 上仍选 `cuda.mma` 的配置，并说明是哪条条件把它挡在 TCGEN05 之外；
3. `M=32` 时 sm_100 上的 TCGEN05 还能用吗？如果可以，`GetTCGEN5MMAMeta` 返回的
   `atom_m` 与 `enable_ws` 是什么？（提示：读 `src/op/tcgen5_meta.h:22-177`）

---

## E10 动手题 ★★ 不需要硬件

用 `code/03-tma-descriptor/tma_descriptor.py`：

1. 为 shape=(8192, 8192)、dtype=bf16、box=(256, 64) 的 tile 打印出
   `cuTensorMapEncodeTiled` 的全部参数；
2. 解释 `globalStride` 为什么只有 1 项（rank=2），以及它的单位是什么；
3. 把 box 的最内层改成 8，会发生什么？为什么？

---

## E11 概念题 ★★ 不需要硬件

对照 `_vendor/tilelang-v0.1.15/examples/warp_specialize/example_warp_specialize_gemm_copy_0_gemm_1.py`：

1. `T.ws(0)` 与 `T.ws(1)` 分别展开成什么条件？
2. 两个 mbarrier 的 `arrive_count=128` 为什么是 128 而不是 256？
3. 如果去掉 producer 分支里的 `T.barrier_wait(compute_is_done, (ko + 1) % 2)`，
   会发生什么类型的错误？为什么要用 `(ko + 1) % 2` 而不是 `ko % 2`？
4. 这一整套变换在编译期由哪三个 pass 完成？它们的前置门禁是什么？

---

## E12 概念题 ★★ 不需要硬件

关于 Blackwell 的 TMEM：

1. TMEM 的物理尺寸是多少？分配单位与列数有什么约束？（引用 PTX ISA 文档位置）
2. 有哪两类操作可以访问 TMEM？这对 kernel 的结构意味着什么？
3. 要让 `T.gemm` 在 sm_100 上选 `tcgen05`，`C` 必须满足什么作用域条件？
4. `T.gemm_blockscaled` 在 SM100 与 SM120 上分别走哪条指令？两者对 `mbar` 的要求有何不同？

---

## E13 概念题 ★★ 不需要硬件

关于 cluster 与 DSM：

1. `T.Kernel(..., cluster_dims=(4,1,1))` 与 `T.ClusterKernel(..., cluster_dims=(4,1,1))`
   有区别吗？
2. 生成代码里，带 cluster 的启动为什么要额外调用
   `cudaFuncSetAttribute(..., cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`？
   厂商文档对可移植 cluster 上限怎么说？
3. `T.copy_cluster` 的“TMA 快路径 / Multi-TMA / SIMT 回退”分别在什么条件下触发？
   各自的 mbarrier arrive 数是多少？

---

## E14 概念题 ★★ 不需要硬件

关于 CuTe DSL 后端：

1. TileLang 的 CuTe DSL 后端复用了哪一条 pass pipeline？在哪一行可以看出来？
2. 它生成的 device 代码是什么形态？由谁最终编译成 cubin？
3. 它与 NVIDIA 官方的 cuTile Python 是同一件事吗？如果不是，请说出两者在
   “编译目标”上的区别，并给出 corpus 里的依据位置。

---

## E15 动手题 ★★★ 不需要硬件（可选：需要 GPU 才能运行验证）

关于版本差异与 API 可用性：

1. 用 `diff -rq` 比较 `_vendor/tilelang-v0.1.14/` 与 `_vendor/tilelang-v0.1.15/` 的
   `src/cuda` 与 `tilelang/cuda` 两个目录，列出与 NV codegen 有关的差异文件；
2. 至少举出三条具体变化，并给出对应的源码位置；
3. 核对 `T.tcgen05_gemm` 是否在 `tilelang.language` 的导出名单里（读
   `tilelang/cuda/language/__init__.py` 与 `tilelang/language/__init__.py`），
   并解释为什么“官方示例里能写”与“facade 的 `__all__` 里没有”会同时成立。
