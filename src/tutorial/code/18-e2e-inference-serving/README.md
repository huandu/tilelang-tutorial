# 第 18 章 代码

本章是端到端的推理与部署章，代码分三类：

1. **纯标准库的账本与仿真**（`01-` 到 `05-`，以及 `07-` 里的
   `moe_dispatch_contract.py`）——不需要 GPU / NPU，也不依赖 TileLang 与
   PyTorch。它们在写作机上**实际运行过**，输出抄在各目录的 README 里。
2. **TileLang kernel 骨架**（`06-` 与 `07-` 里的 `skeletons.py`）——需要真实
   NVIDIA GPU 才能编译运行。本章只做到**仅静态核对**：用
   `python3 -m py_compile` 过语法，逐个在 `_vendor/` 里比对 API 名与签名。
   **没有在硬件上运行过任何一条。**
3. **昇腾 ACLGraph 示例**（`08-`）——需要昇腾 NPU + CANN + torch_npu +
   TileLang-Ascend 适配器；验证级别同样是**仅静态核对**，并且明确标注它属于
   适配器仓库的 API（不是主干 v0.1.15 的昇腾方言）。

## 目录

```text
code/
├─ README.md                              本文件
├─ 01-prefill-decode-roofline/            纯标准库；prefill / decode 的算术强度与权重底线
├─ 02-kv-ledger/                          纯标准库；Qwen2.5-7B 的 KV 账本与 paged 块账
├─ 03-decode-step-share/                  纯标准库；一次 decode step 的算子耗时占比（本章第 4 节的表）
├─ 04-serving-schedule-sim/               纯标准库；static 与 continuous batching 的调度仿真
├─ 05-bench-methodology/                  纯标准库；TTFT/TPOT 口径换算与污染检查
├─ 06-tilelang-decode-attention/          TileLang；decode attention 的 split-KV 骨架
├─ 07-tilelang-moe-and-quant/             纯标准库 + TileLang；MoE dispatch 契约与量化 GEMM 骨架
└─ 08-ascend-aclgraph/                    昇腾；ACLGraph 入图 + 一步 decode 的骨架
```

每个子目录都能独立复制出去运行，不依赖其他章节或其他示例的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-prefill-decode-roofline/prefill_decode_roofline.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 |
| `02-kv-ledger/kv_ledger.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `03-decode-step-share/decode_step_share.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `04-serving-schedule-sim/serving_schedule_sim.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `05-bench-methodology/bench_methodology.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `06-tilelang-decode-attention/decode_attention_splitkv.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**（`py_compile` + API 比对）；未运行 |
| `07-tilelang-moe-and-quant/moe_dispatch_contract.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过，三项断言全部通过 |
| `07-tilelang-moe-and-quant/skeletons.py` | 同 `06-` | **仅静态核对**；未运行 |
| `08-ascend-aclgraph/aclgraph_decode_block.py` | 昇腾 NPU + CANN + torch_npu + `_vendor/tilelang-ascend-v0.1.2.000` | **仅静态核对**；未在 NPU 上运行 |

## 统一说明

* 写作环境（见 [`../../VERSIONS.md`](../../VERSIONS.md)）：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章**没有任何一条本机实测的性能或正确性结果**。所有需要在加速器上
  运行的示例，预期输出一律写成「正确性判据 + 性能口径」。
* 凡出现的数字只有三类来源：**本书模型输出**（可在任意机器复算，脚本在
  `code/` 里）、**厂商或上游资料的既有数字**（带 corpus 路径与测量条件）、
  **仿真输出**（参数人为设定，脚本里写明）。三类都在正文与 README 里标了。
* 版本一律锚定到 TileLang `v0.1.15` 与 TileLang-Ascend
  `v0.1.2.000-release` 这两个 tag（见 `../../VERSIONS.md`）；
  本章不写「跟随上游最新」这类无法复核的模糊说法。
