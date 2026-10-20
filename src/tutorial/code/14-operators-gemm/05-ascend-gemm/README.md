# 示例 05：昇腾上的 GEMM 与 batched GEMM

## 目的

把第 14 章第 6 节的昇腾对照写成可运行代码，三个实现分别对应本章阶梯的不同
级别：

| 函数 | 对应本章概念 | 关键差异 |
| --- | --- | --- |
| `ascend_gemm_basic` | 第 ② 级 | `T.alloc_L1` / `T.alloc_L0C` / `T.gemm_v0` / 显式 `T.barrier_all()` |
| `ascend_batch_gemm` | 4.4 节 | 一维 grid 展平 `(batch, m_block, n_block)`，用 `pass_configs` 打开自动同步/多缓冲/内存规划 |

## 运行前提

* 昇腾 NPU + CANN（含 `bisheng` 与 Ascend runtime）+ `torch_npu`。
* TileLang-Ascend：`_vendor/tilelang-ascend-v0.1.2.000`（`pip install -e .`）。
  该仓库的 `T.alloc_L1` / `T.alloc_L0C` / `T.gemm_v0` 在
  `tilelang.language` 下，与本示例的 import 一致。
* **注意**：主干 `tilelang-v0.1.15` 的昇腾方言走的是
  `T.alloc_l1`（小写）/ `T.gemm`（带 `unit_flag_ctrl`）/ `T.Persistent`
  这套 API，两个仓库的写法不通用。本示例用的是**适配器仓库**的 API。

## 运行命令

```bash
# 基础 GEMM + batched GEMM
python3 ascend_gemm.py --m 1024 --n 1024 --k 1024 --batch 8
```

## 期望输出

```
ascend_gemm_basic: 正确性通过
ascend_batch_gemm: 正确性通过
```

* **正确性判据**：与 `a @ b` / `torch.matmul(a, b)` 比较，
  `rtol=1e-2, atol=1e-2` 通过。
* **性能口径**：本示例不测性能。若要与 CATLASS / 手写基线对齐，请看
  `../references.md` 第 6.3 节与 `../06-bench-methodology/README.md` 的清单，
  并至少固定：基线、shape、dtype、epilogue、测量协议五件事。
* **本机没有 NPU，以上输出均未产生。** 作者实际做到的是静态核对：
  `py_compile` 通过；`T.alloc_L1`、`T.alloc_L0C`、`T.gemm_v0`、`T.Scope`、
  `T.barrier_all`、三个 `PassConfigKey` 逐个在
  `_vendor/tilelang-ascend-v0.1.2.000/` 中比对过。

## 已知局限

* `ascend_gemm_basic` 的 `block_M=128, block_N=256, K_L1=64` 来自官方示例的
  默认值，没有针对具体 shape 调优。
* 与 NV 版本不同，昇腾侧的 `M`、`N` 必须能被 tile 整除（官方示例直接
  `M // block_M`，示例本身不处理 partial tile）；尾块处理见适配器的
  `example_gemm_tail_block_developer.py`。
* 昇腾的 `T.gemm_v0` 是适配器仓库的接口；主干 v0.1.15 用 `T.gemm` +
  `unit_flag_ctrl`，并支持 `T.Persistent` 的 persistent 执行模型。
