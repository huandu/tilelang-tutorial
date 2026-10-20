# 第 34 章 代码

本章讲的是**不是矩阵乘**的那一类算子：Engram 的门控/寻址，以及 mHC 的
mix 拆分与 Sinkhorn 归一化。它们全部来自 DeepSeek 2026-09-30 开源的
[TileKernels](https://github.com/deepseek-ai/TileKernels)（corpus 快照：
`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/`）。

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU、没有昇腾 NPU**
（见 `../../VERSIONS.md`），所以本章的 kernel 全部只做**静态阅读**，`code/` 下
能跑的都是**在 CPU 上重写同一套数学**的脚本——它们验证的是“形状与代数”，
不是“kernel 跑出来的结果”。

## 目录

```
code/
├─ README.md                    本文件
├─ 01-engram-gate-shapes/       Engram 门控的 shape 走查 + 融合恒等式（fp32/fp64 CPU 复算）
├─ 02-sinkhorn-iterations/      两种 Sinkhorn：mHC 的 4x4 双随机 与 Engram 的 L2 行列缩放
├─ 03-mhc-mix-flow/             mHC 的 expand / split_mixes / apply_mix / post 数据流
└─ 04-engram-hash-lookup/       Engram 的 n-gram 哈希与 O(1) 查表（确定性寻址）
```

每个子目录都能独立复制出去运行，不依赖其他章节或其他示例的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-engram-gate-shapes/engram_gate_shapes.py` | Python 3.9+、PyTorch（CPU 即可） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11、torch 2.1.1 上实际运行过，输出见该目录 README |
| `02-sinkhorn-iterations/sinkhorn_iterations.py` | 同上 | **CPU 逻辑验证**：运行过 |
| `03-mhc-mix-flow/mhc_mix_flow.py` | 同上 | **CPU 逻辑验证**：运行过 |
| `04-engram-hash-lookup/engram_hash_lookup.py` | 同上 | **CPU 逻辑验证**：运行过 |

**没有任何一行来自本章的代码在 GPU / NPU 上运行过。** 所有 kernel 的行为描述都来自
逐行阅读 corpus 里的源码；`code/` 里的脚本只复现同一套公式与形状约定，用来让读者
能自己核对“这个算子到底在算什么”。

## 与 corpus 源码的对应关系

| 本目录的脚本 | 对应的 corpus 源码 |
| --- | --- |
| `01-engram-gate-shapes` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py`、`.../engram_fused_weight_kernel.py`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py`（`engram_gate_ref`） |
| `02-sinkhorn-iterations` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_kernel.py`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_kernel.py`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py` |
| `03-mhc-mix-flow` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/{pre_split_mixes,pre_apply_mix,post,expand,head_compute_mix}_kernel.py` 与 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py` |
| `04-engram-hash-lookup` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_cuda.py`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py`（`engram_hash_ref`、`make_offsets`）、`.../Engram/engram_demo_v1.py` |

## 你需要在真机上做什么

上面四个脚本都能在没有加速器的机器上跑。想知道这些 kernel 在硬件上的表现，需要：

1. NVIDIA 侧：SM90/SM100 GPU、CUDA ≥ 13.1、PyTorch ≥ 2.13、TileLang ≥ 0.1.15
   （TileKernels `README.md:26-28`）；
2. 昇腾侧：Ascend 950、CANN ≥ 9.2.0（`README.md:29-31`）；
3. 用 TileKernels 自带的 `tests/` 跑正确性与 benchmark：
   `python -m pytest tests/engram/test_engram_gate_fwd.py --run-benchmark`。

本章不给出任何实测数字；需要真机才能得到的数字见 `../references.md` 里标注的口径。
