# 示例 1：三层接口 API 清点表

## 目的

把第 03 章那句最容易变成空话的结论——「beginner / developer / expert 三层各自对应
v0.1.15 里的一组具体构造」——变成一张**可以复算的表**。

脚本不导入 `tilelang`（本机没有 CUDA/NPU，编译好的 `.so` 不存在），而是直接读锁定版本的
源码，在语言层的导出面（`tilelang/language/**`、`tilelang/cuda/language/**`）里逐个确认
每个 API 名字确实存在，并打印它出现的 `文件:行号`。同时核对官方文档里
「beginner 层尚未完全实现」那句说明是否还在。

## 运行前提

* Python 3.9+（只用标准库：`argparse` / `re` / `pathlib`）。
* **不需要** GPU、NPU、CUDA toolkit，也不需要安装 TileLang。
* 需要本书的 `_vendor/tilelang-v0.1.15/` 存在（脚本会自动从自身位置向上寻找书根目录）。

## 运行命令

```bash
# 在本书根目录下
python3 chapters/03-design-philosophy/code/01-layer-inventory/layer_inventory.py

# 或从示例目录里直接跑（脚本自己找书根）
cd chapters/03-design-philosophy/code/01-layer-inventory
python3 layer_inventory.py

# 只想看缺了谁
python3 layer_inventory.py --list-missing
```

## 期望输出

作者在写作机上的实际输出（2026-10-01，锁定版本 `a35f8dd`，共 62 个语言层源文件）：

```
== beginner (eager 前端，层本身未完成) ==
  [有]   const                  def     tilelang/language/eager/builder.py:1177
  [有]   dynamic                def     tilelang/language/symbolics.py:13
  [有]   empty                  def     tilelang/language/allocate.py:397
  [有]   Tensor                 export  tilelang/language/common.py:209
  [有]   StridedTensor          export  tilelang/language/common.py:208
  [有]   macro                  def     tilelang/language/eager/builder.py:287
  [有]   prim_func              def     tilelang/language/ast/ir.py:171

== developer (tile 库) ==
  [有]   Kernel                 def     tilelang/cuda/language/kernel.py:13
  [有]   Parallel               def     tilelang/cuda/language/loop.py:18
  [有]   Pipelined              def     tilelang/language/loop.py:112
  [有]   Persistent             def     tilelang/language/loop.py:82
  [有]   copy                   def     tilelang/cuda/language/copy_op.py:19
  [有]   gemm                   def     tilelang/cuda/language/gemm_op.py:32
  [有]   gemm_sp                def     tilelang/cuda/language/gemm_op.py:172
  [有]   gemm_blockscaled       def     tilelang/cuda/language/gemm_op.py:84
  [有]   reduce                 def     tilelang/language/ast/ir.py:564
  [有]   reduce_max             def     tilelang/cuda/language/reduce_op.py:24
  [有]   reduce_sum             def     tilelang/language/reduce_op.py:230
  [有]   cumsum                 def     tilelang/language/scan_op.py:88
  [有]   cummax                 def     tilelang/language/scan_op.py:175
  [有]   fill                   def     tilelang/language/fill_op.py:11
  [有]   clear                  def     tilelang/language/fill_op.py:48
  [有]   alloc_shared           def     tilelang/language/allocate.py:49
  [有]   alloc_fragment         def     tilelang/language/allocate.py:81
  [有]   annotate_layout        def     tilelang/language/annotations.py:45
  [有]   use_swizzle            def     tilelang/language/annotations.py:30
  [有]   GemmWarpPolicy         export  tilelang/language/common.py:197

== expert (thread 原语与内建) ==
  [有]   get_thread_binding     def     tilelang/language/kernel.py:237
  [有]   get_block_binding      def     tilelang/language/kernel.py:260
  [有]   alloc_local            def     tilelang/language/allocate.py:67
  [有]   alloc_var              def     tilelang/language/allocate.py:96
  [有]   serial                 def     tilelang/language/ast/ir.py:676
  [有]   vectorized             def     tilelang/language/ast/ir.py:732
  [有]   unroll                 def     tilelang/cuda/language/loop.py:57
  [有]   sync_threads           def     tilelang/language/builtin.py:963
  [有]   atomic_add             def     tilelang/cuda/language/atomic.py:12
  [有]   tvm_thread_allreduce   export  tilelang/language/ast/ir.py:2126
  [有]   comm_reducer           def     tilelang/language/ast/ir.py:1636
  [有]   attr                   def     tilelang/language/ast/ir.py:1050
  [有]   evaluate               def     tilelang/language/ast/ir.py:1316
  [有]   ptx_wait_group         export  tilelang/language/ast/ir.py:2149

== CUDA 方言扩展（只有 CUDA target 才有） ==
  [有]   ClusterKernel          def     tilelang/language/kernel.py:458
  [有]   copy_cluster           def     tilelang/language/copy_op.py:133
  [有]   mbarrier_wait_parity   def     tilelang/language/builtin.py:457
  [有]   tma_load               def     tilelang/language/builtin.py:321
  [有]   ldg128                 def     tilelang/language/builtin.py:1685
  [有]   alloc_tmem             def     tilelang/language/allocate.py:254

== 官方文档核对：beginner 层的实现状态 ==
  [有] docs/get_started/overview.md:20  含 “not yet fully implemented”

合计 47 个名字，找到 47 个，缺失 0 个。
```

输出里的 `def` / `export` 两列说明这条证据是什么：`def` 表示找到了函数定义，`export`
表示只在导出清单（`__all__`）里找到了这个名字——例如 `Tensor`、`GemmWarpPolicy`
这类对象是类的再导出，不是函数。

**判据**：退出码为 `0`，且「缺失 0 个」。若某个名字找不到，退出码为 `1`，说明本章的
三层清单与锁定版本已经对不上，需要重新核对。

## 注意事项

* 脚本证明的是「这个名字出现在语言的导出面上」，**不是**「它在你的环境里能跑通」。
  后者需要在装了匹配 CUDA/驱动或昇腾 CANN 的机器上实际编译运行。
* 名字出现的位置是**首次出现**处，可能是 `__all__` 清单、`def` 定义或某个方言的
  再导出，不一定是最原始的定义文件。下一节（第 04 章）会给出更细的模块地图。
* 本示例不涉及昇腾专属构造（`T.alloc_l1` / `T.alloc_l0c` / `T.dual_copy` 等）——
  它们属于 `tilelang/language/ascend` 与 `tilelang-ascend` 适配器，见第 19、25 章。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0。
未在真实加速器上运行过任何 kernel。
