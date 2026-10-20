# 第 17 章 代码

本章代码围绕「从零训练一个 7B 模型」的里程碑组织。**写作机器没有 NVIDIA GPU，也没有
昇腾 NPU**，所以每个示例都明确标注了它能达到的最高验证级别：

| 目录 | 目的 | 是否需要加速器 | 验证级别 |
| --- | --- | --- | --- |
| `01-plan-ledger/` | 7B 的显存与算力账本估算 | 否（纯 Python） | **CPU 逻辑验证**（已运行，输出见正文 §2） |
| `02-data-pipeline/` | 流式读取 → 去重 → packing → shuffle | 否（离线模式） | **CPU 逻辑验证**（已运行） |
| `03-tilelang-norm-torch/` | RMSNorm 的 TileLang 前向 + autograd 接入 | 是（CUDA / NPU） | 仅静态核对 |
| `04-verify-numerics/` | kernel 与 torch 参考的数值对齐验收 | 是（CUDA） | 仅静态核对 |
| `05-ascend-integration/` | `torch_npu` + `torch.ops.tl_ascend` 接入 | 是（昇腾 NPU） | 仅静态核对 |
| `06-train-loop/` | 预训练循环（混合精度/梯度累积/ckpt/恢复） | 冒烟模式不需要 | **CPU 逻辑验证**（已运行）；`--from-hf` 路径仅静态核对 |

## 统一约定

* 所有 `.py` 都能通过 `python3 -m py_compile <file>`（写作者已逐个执行）。
* 涉及 TileLang import 的示例，API 名与签名都在 `_vendor/tilelang-v0.1.15/` 与
  `_vendor/tilelang-ascend-v0.1.2.000/` 里逐项核对过（见本章 `references.md`）。
* 「仅静态核对」**不等于**「运行验证」。要下结论，请在有卡的机器上按各目录
  README 的命令重跑。
* 代码里不出现任何私有路径、token 或 IP。

## 依赖

* CPU 可跑的示例：Python 3.11、`datasets`、`transformers`、`torch`（`01/02` 只需要标准库）。
* 需要 CUDA 的示例：可用 `tilelang`（v0.1.15）、NVIDIA GPU。
* 需要昇腾的示例：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1、TileLang-Ascend
  `v0.1.2.000-release`。
