# 示例 05：GEMM + ReLU 混核 kernel（M 树 / 主干方言）

## 目的

把本章 §7 的 M 树方言落成可运行代码，同时演示三件 A 树没有的写法：

1. `T.alloc_l1` / `T.alloc_l0c`（小写）；
2. `T.Persistent` + `T.Pipelined` 的自动调度；
3. `T.dual_copy` + `T.SimtVF`：Cube 的结果经 **L0C → UB → GM**，在 Vector 上做 ReLU。

来源：`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:66-108` 的快速开始。

## 运行前提

* Ascend 950 NPU + 兼容 CANN（含 `bisheng`、CCE 版 `ld.lld`、Ascend runtime）+ `torch_npu`；
* 主干 TileLang `v0.1.15`（`USE_ASCEND=ON USE_CUDA=OFF` 安装）；
* `M % block_M == 0`、`N % block_N == 0`、`K % block_K == 0`（本片段不处理尾块）。

## 运行命令

```bash
python3 gemm_relu_dialect.py --m 256 --n 7168 --k 2048
python3 gemm_relu_dialect.py --m 256 --n 7168 --k 2048 --bench
```

## 期望输出

```
GEMM + ReLU passed.
```

带 `--bench` 时额外打印 `gemm_relu (M=..., N=..., K=...): ... us | ... TFLOPS`
（该数字来自你机器上的 msProf，**本书不提供任何具体数值**）。

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`alloc_l1`、`alloc_l0c`、`alloc_shared`、`T.Persistent`、
`T.Pipelined`、`T.gemm`（含 `transpose_B`/`clear_accum`）、`T.dual_copy`、`T.SimtVF`、
`l2_cache_ctrl="NOTALLOC_KEEP"/"NOTALLOC_PW"` 逐个在
`_vendor/tilelang-v0.1.15/` 中比对过（`allocate.py:26,55`、`copy_op.py:53`、
`frame.py:107`、`gemm_op.py:168`、`README.md:66-108`）。

## 已知局限

* `block_N=224`、`num_blocks=32`、L2 提示的取值都照抄自官方 quickstart，
  **没有针对具体 shape 重新调优**。
* 本示例只覆盖 M 树；A 树的同款写法见 `../04-gemm-l1-l0c/`，两者不可混用。
