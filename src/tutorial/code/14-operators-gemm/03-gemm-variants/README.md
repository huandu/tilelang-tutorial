# 示例 03：GEMM 的四个变体

## 目的

把第 14 章第 4 节的四个变体放在同一个文件里，方便对照它们的差异：

| 函数 | 变体 | 判据 |
| --- | --- | --- |
| `gemm_splitk` | split-K | 输出块数 < 核数，且 K 可以切 |
| `gemm_streamk` | stream-K | 输出块数够多但不是核数整数倍，尾波不满 |
| `gemm_grouped` | grouped GEMM | 多个形状不一致的 GEMM（MoE） |
| `gemm_batched` | batched GEMM | 多个形状一致的 GEMM |

另外提供 `streamk_plan(...)`：一个**纯 Python** 的函数，不需要 GPU 就能算出
stream-K 的规模参数（`total_tiles` / `streamk_tiles` / `blocking_tiles` /
`full_tiles` / `partial_tiles` / `sm_partition_factor`）。它的公式逐行照搬
官方 `example_tilelang_gemm_streamk.py`。

## 运行前提

* NVIDIA GPU + CUDA toolkit + PyTorch with CUDA + TileLang `v0.1.15`。
* `python3 gemm_variants.py` 会跑 split-K 与 batched 的正确性检查，并打印
  stream-K 的规划结果。

## 运行命令

```bash
python3 gemm_variants.py
```

## 期望输出

```
split-K 正确性通过
batched 正确性通过
stream-K plan: {...}
```

* **正确性判据**：split-K 与 `(a @ b).float()` 比、batched 与 `torch.bmm` 比，
  `rtol=1e-2, atol=1e-2` 通过。
* **性能口径**：本示例**不测性能**。如果想测，请按 `../06-bench-methodology/`
  的协议补上 `warmup` / `rep` / `backend` / `return_mode`。
* **本机没有 GPU，以上输出均未产生。** 作者实际做到的是静态核对：
  `py_compile` 通过；`T.atomic_add`、`T.alloc_fragment((1,), T.int32, "local")`、
  `T.if_then_else`、`T.alloc_var`、`T.Persistent`/`T.Pipelined` 的用法逐个与
  `_vendor/tilelang-v0.1.15/` 的官方示例比对过。

## 已知局限

* `gemm_streamk` 只保留「K 迭代平均分配 + 被切块原子合并 + 整块平铺」这条主干，
  省略了官方示例里的 2-CTA / persistent 调度与 `order` 优化。
* `gemm_grouped` 用的是偏移表写法；官方另有指针表写法
  （`example_grouped_gemm_fwd_ptr.py`），但那条路径的**多级流水尚不正确**，
  所以这里不提供。
* 四个变体都**没有**调优过 tile 尺寸；它们演示的是「结构」与「判据」，不是
  某个 shape 上的最优实现。
