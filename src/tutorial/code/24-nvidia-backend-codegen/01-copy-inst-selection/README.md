# 01 · 复现 `T.copy` 的搬运指令选择

## 目的

把 `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-891` 的
`SelectCopyInstForLowering` 判定顺序，用标准库重写成一个可交互的小模型，用来回答：

* 为什么同样一句 `T.copy`，在 sm_80 和 sm_90 上会落到不同机制？
* “默认路径下只有 store 会自动走 TMA”这句话具体长什么样？
* 哪些条件会让 `T.copy(A_g, A_s, prefer_instruction="tma")` 编译失败？

## 运行前提

Python 3.9+，只用标准库。**不需要 GPU，也不需要安装 TileLang。**

## 运行命令

```bash
python3 copy_inst_selection.py
```

## 期望输出

脚本打印一份场景表，每个场景一行判定结果；最后做一次自检（与下面列出的期望值逐条
比对，不一致就返回非零退出码）。参考片段：

```text
sm_90        shared.dyn -> global         bits=16  last=64   hint=default    -> BulkStore
sm_90            global -> shared.dyn     bits=16  last=64   hint=default    -> Normal
sm_90            global -> shared.dyn     bits=16  last=64   hint=pipeline   -> CPAsync
sm_90            global -> shared.dyn     bits=16  last=64   hint=tma        -> BulkLoad
sm_80            global -> shared.dyn     bits=16  last=64   hint=default    -> Normal
sm_90            global -> shared.dyn     bits=16  last=30   hint=tma        -> ERROR: ...
sm_90            global -> shared.dyn     bits=16  last=64   hint=cluster    -> BulkLoad
self-check: ok
```

三条最值得注意的：

1. **`store` 默认走 TMA，`load` 默认不走**（`hint=default` 的两行）。
2. **裸 `T.copy(A_g, A_s)` 的兜底是同步 SIMT（`Normal`），不是 `cp.async`**；
   `cp.async` 只在“流水里”或“显式 `T.async_copy`”时出现。
3. sm_80 上没有 TMA，所以即使是最内层 16 字节对齐的 2D tile 也只能走
   `Normal` / `cp.async`。

## 作者实际做到的验证级别

**CPU 逻辑验证**：本脚本在 macOS (Apple Silicon)、Python 3.11 上运行过，输出与
README 中列出的期望判定一致。它没有、也无法验证编译器在真实硬件上的行为。

## 与源码的对应关系

| 本脚本 | `copy_analysis.cc` |
| --- | --- |
| `select_copy_inst()` 的分支顺序 | `SelectCopyInstForLowering`（:716-807） |
| `check_bulk_load/store()` | `CheckBulkLoad` / `CheckBulkStore`（:195-261） |
| `check_global_strides()` | `CheckGlobalStrides`（:143-193） |
| `target_has_bulk_copy()` | `TargetHasBulkCopy`（`_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:117-122`） |
| `target_has_async_copy()` | `TargetCudaHasAsyncCopy`（`_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:85-90`） |

## 简化了什么

1. 不实现 `arith::Analyzer`：所有形状都是具体整数，不做符号证明。
2. 不实现 1D TMA 路径的完整布局检查（`IsSemanticallyLinearLayout`），只检查连续性与对齐。
3. 不实现 gather4 / scatter4、im2col、FP4 的 dtype 特例，只保留主判定链。
4. `ldmatrix` / `stmatrix` / TMEM 的判断用 scope 字符串近似（真实实现用
   `TargetHasLdmatrix` / `TargetHasStmatrix` / `TargetHasTmem` 加 `IsTmemBuffer`）。
