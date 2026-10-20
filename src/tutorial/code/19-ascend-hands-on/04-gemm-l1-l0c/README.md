# 示例 04：昇腾最小 GEMM（A 树）

## 目的

本章 §5 的「第一个昇腾 GEMM」。展示 `GM → L1 → L0A/L0B → Cube → L0C → GM` 的完整
通路，并说明 `T.gemm_v0` 这一行背后展开成了多少件事（L1→L0 搬运、Cube MAD、L0C 写回）。

## 关键 API（逐个在 A 树里核对过）

| API | 位置 | 说明 |
| --- | --- | --- |
| `T.alloc_L1(shape, dtype)` | `tilelang/language/allocate.py:140` | scope `shared.l1` |
| `T.alloc_L0C(shape, dtype)` | `tilelang/language/allocate.py:152` | scope `wmma.accumulator` |
| `T.gemm_v0(A, B, C, transpose_A, transpose_B, init, kL0Size, n_actual)` | `tilelang/language/ascend.py:418` | docstring 明确 A/B 在 L1、C 在 L0C |
| `T.barrier_all()` | `tilelang/language/ascend.py:275` | 全 PIPE 屏障 |

## 运行前提

* 昇腾 NPU + CANN + `torch_npu` + TileLang-Ascend（A 树）；
* M 能被 128 整除、N 能被 256 整除、K 能被 64 整除（本示例不处理尾块）。

## 运行命令

```bash
python3 gemm_adapter.py --m 1024 --n 1024 --k 1024            # fp16
python3 gemm_adapter.py --m 1024 --n 1024 --k 1024 --fp32      # fp32
```

## 期望输出

```
init successful!
Kernel Output Match!
```

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`alloc_L1`、`alloc_L0C`、`gemm_v0`、`Scope`、
`barrier_all` 逐个在 `_vendor/tilelang-ascend-v0.1.2.000/` 中比对过；
对照官方 `examples/gemm/example_gemm.py` 逐行核过。

## 已知局限

* 两次全屏障换正确性，**没有流水重叠**；高性能版请看官方 `example_gemm_intrinsic.py` /
  `example_gemm_persistent.py`。
* `block_M=128, block_N=256` 的 fp32 累加正好占满 L0C（128 KB），再大需要靠 `kL0Size`
  调整 L0 双缓冲。
* 昇腾侧 M/N 必须配合 tile 整除；官方示例同样如此。
