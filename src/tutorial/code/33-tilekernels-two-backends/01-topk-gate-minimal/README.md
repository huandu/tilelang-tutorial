# 示例 01：top-k gate 的两个后端最小可读版

## 目的

正文 §4.2 与 §4.3 的对照对象。把 TileKernels 的
`moe/topk_gate_cuda.py`（68 行）与 `moe/topk_gate_asc.py`（157 行）各自精简成
**能一眼看完、但结构完整**的版本，用来回答一个问题：

> 同一个算法，用同一门 DSL 写在两块芯片上，差异到底在哪几行？

两个文件都**不是**可以直接跑生产的版本（没有完整的形状校验、没有 PDL、
Ascend 版只留了单组快路径），它们是**阅读材料**。

## 与语料的关系

| 本文件 | 改写自 | 保留 | 省略 |
| --- | --- | --- | --- |
| `topk_gate_cuda_min.py` | `TileKernels/tile_kernels/moe/topk_gate_cuda.py` | 一行一 block、`T.alloc_fragment`、`T.alloc_reducer(op='min')`、`T.Parallel`、`disable_tma=True`、`TL_DISABLE_WARP_SPECIALIZED` | 非有限值检查（`:40-46`）、PDL（`:23-24,65-66`） |
| `topk_gate_asc_min.py` | `TileKernels/tile_kernels/moe/topk_gate_asc.py` | `T.Kernel(num_vec_cores)`、UB 上的 `T.alloc_shared`、`T.annotate_buffer_versions`、`T.Pipelined(..., annotations={'multi_buffer_eligible': ...})`、`T.view`、`T.SimdVF()`、`vector_length = 64`、`S.vcmax`/`S.vcmin`、收尾断言 | 多组慢路径（`:106-151`）、可选 cache 提示 |

改写时把“哪一行在保护什么”写成注释，行号引用见 `../references.md`。

## 运行前提

* `topk_gate_cuda_min.py`：NVIDIA SM90 或 SM100、CUDA Toolkit ≥ 13.1、
  TileLang `v0.1.15`（`import tilelang.language as T`，CUDA 方言）、PyTorch。
* `topk_gate_asc_min.py`：Ascend 950 NPU、CANN ≥ 9.2.0、`torch_npu`、
  TileLang `v0.1.15`（`from tilelang.ascend import language as T`，
  **主干 M 树的昇腾方言**，不是第 19 章的适配器 A 树）。

> ⚠️ 两个文件的 `import` 指向**不同方言**，不要放进同一个进程。
> `topk_gate_asc_min.py` 只支持 `num_groups == 1`（文件里有断言）。

## 运行命令

```bash
# CUDA 机器上
python3 -c "
import torch
from topk_gate_cuda_min import topk_gate
s = torch.randn(512, 72, dtype=torch.float32, device='cuda')
print(topk_gate(s, 6).shape)          # 期望 torch.Size([512, 6])
"

# Ascend 机器上
python3 -c "
import torch, torch_npu
from topk_gate_asc_min import topk_gate
s = torch.randn(512, 72, dtype=torch.float32, device='npu')
print(topk_gate(s, 6, num_vec_cores=2 * torch.npu.get_device_properties(0).cube_core_num).shape)
"
```

## 期望输出

```
torch.Size([512, 6])
```

**正确性判据**（在有硬件时）：与 `tile_kernels/torch/topk.py:18-21` 的
`stable_topk` **逐字节相同**（含 tie-break，比对方式见
`tile_kernels/testing/numeric.py:18-21`）。**性能口径**：本示例不测性能。

**本机没有 GPU / NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile` 通过；`T.alloc_reducer` / `reducer_init` /
`reducer_update` / `finalize_reducer` / `T.annotate_buffer_versions` /
`T.view` / `T.SimdVF` / `T.Pipelined` / `S.vcmax` / `S.vcmin` /
`S.vsel` / `S.vsts` / `S.pset` / `S.vci` 逐个在 `_vendor/tilelang-v0.1.15/` 中
比对过**存在性与签名**。**没有在任何硬件上运行过。**

## 已知局限

* Ascend 版只保留单组快路径；`num_experts` 超过 `6 × 64 = 384` 时原生实现会走
  多组慢路径。
* CUDA 版省略了非有限值检查，所以喂 `NaN` 时行为与原生实现不同。
* 两个文件都假设输入是连续的二维 fp32 张量。
