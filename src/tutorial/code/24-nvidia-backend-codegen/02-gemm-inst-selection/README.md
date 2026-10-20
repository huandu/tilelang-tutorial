# 02 · 复现 `T.gemm` 的指令选择

## 目的

把 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375` 的 `Gemm::SelectInst`
谓词链，以及 `_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h:22-177` 的
`GetTCGEN5MMAMeta` 判定，用标准库重写成可交互模型。用来回答：

* 同样一句 `T.gemm`，在 sm_70 / sm_75 / sm_80 / sm_90 / sm_100 上分别选什么？
* 「我的 GEMM 为什么没用 TCGEN05」——是哪一条不满足？
* fp8 的 WGMMA 为什么对 transpose 这么挑？

## 运行前提

Python 3.9+，只用标准库。**不需要 GPU，也不需要安装 TileLang。**

## 运行命令

```bash
python3 gemm_inst_selection.py
```

## 期望输出

脚本打印一张「配置 → 指令」表，最后做自检。参考片段：

```text
sm_90a  M=128 N=128 K=64   f16->f32  A=shared B=shared C=fragment  -> cuda.wgmma
sm_90a  M=32  N=128 K=64   f16->f32  A=shared B=shared C=fragment  -> cuda.mma
sm_100  M=128 N=128 K=64   f16->f32  A=shared B=shared C=tmem      -> cuda.tcgen05
sm_100  M=128 N=128 K=64   f16->f32  A=shared B=shared C=fragment  -> cuda.mma
sm_75   M=16  N=16  K=16   f16->f32  A=shared B=shared C=fragment  -> cuda.mma
sm_70   M=16  N=16  K=16   f16->f16  A=shared B=shared C=fragment  -> cuda.mma
sm_70   M=16  N=16  K=16   f16->f32  A=fragment B=shared C=fragment -> cuda.fma
self-check: ok
```

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过。

## 简化了什么

1. 只实现 `GetTCGEN5MMAMeta` 里 **f16/bf16 → f32** 那一支。fp8/fp6/fp4、
   `kind::mxf8f6f4`、int8 等分支没有实现（它们的结构相同，只是 atom 取值不同）。
2. 不实现 `T.gemm` 的 warp 划分（`ComputeWarpPartition`），只回答「选哪条指令」。
3. `CheckWgmma` 只覆盖 f16、bf16、tf32、fp8、int8 的 dtype/转置/K 整除规则；
   B 必须在 shared 这一条也实现了。
4. 不实现 `use_2cta` 之后的 `LowerBlackwell2SM` 等 pass 行为。

## 与源码的对应关系

| 本脚本 | `gemm.cc` / `tcgen5_meta.h` |
| --- | --- |
| `select_inst()` | `Gemm::SelectInst`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375`） |
| `check_wgmma()` | `CheckWgmma`（:40-78） |
| `allow_wgmma()` | `AllowWgmma`（:89-97） |
| `allow_tcgen5_mma()` | `AllowTcgen5Mma`（:80-87） |
| `tcgen5_meta_f16()` | `GetTCGEN5MMAMeta`（`_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h:22-177`） |
| `allow_volta_mma()` / `allow_turing_mma()` | :99-117 / :131-145 |
