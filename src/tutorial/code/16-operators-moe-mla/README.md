# 第 16 章 代码

本章的代码分两类，目的不同：

1. **可以在 CPU 上跑的模型与检查**（`01-`、`02-`、`03-`、`04-`、`05-`、`06-`）。
   `01-` 是 MoE 与 MLA 的成本账本；`02-` 把路由（gating → top-k → 重排 →
   grouped GEMM 分块 → combine）走一遍，验证账本里那个“重排”是真的可逆；
   `03-` 算 KV cache 与 decode 的算术强度；`04-`/`05-`/`06-` 把本章引用到的
   官方 example 逐个扫一遍，核对 API 名与关键常量在锁定版本里是否存在。
2. **只做静态核对的 TileLang 骨架**（没有单独目录）：本章不重复第 13–15 章
   已经给过的算子骨架，而是直接引用锁定版本里的官方 example。凡是引用到的
   kernel 代码，都保证 API 名在 `_vendor/` 里核对过，并标出文件与行号。

## 目录

```text
code/
├─ README.md                        本文件
├─ 01-cost-ledger/
│  ├─ README.md
│  └─ decode_step_ledger.py         纯标准库；MoE / MLA 的成本账本与总结表
├─ 02-moe-routing/
│  ├─ README.md
│  └─ routing_sim.py                纯标准库；路由 + 重排 + 分块 + combine 的可逆性验证
├─ 03-mla-kv/
│  ├─ README.md
│  └─ kv_budget.py                  纯标准库；MHA / GQA / MLA 的 KV 预算与算术强度
├─ 04-moe-kernel-map/
│  ├─ README.md
│  └─ kernel_map.py                 纯标准库；扫 _vendor 官方 example，核对 API 与常量
├─ 05-sparse-index-map/
│  ├─ README.md
│  └─ sparse_index_map.py           纯标准库；NSA / DSA / lightning indexer 的结构对照
└─ 06-ascend-map/
   ├─ README.md
   └─ ascend_map.py                 纯标准库；昇腾 side 的 fusedmoe / permute / dispatch-combine 对照
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-cost-ledger/decode_step_ledger.py` | Python 3.9+ | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，输出见该目录 README |
| `02-moe-routing/routing_sim.py` | Python 3.9+ | **CPU 逻辑验证**：实际运行过，round-trip 断言通过 |
| `03-mla-kv/kv_budget.py` | Python 3.9+ | **CPU 逻辑验证**：实际运行过 |
| `04-moe-kernel-map/kernel_map.py` | Python 3.9+；核对功能需要 `_vendor/` 存在 | **CPU 逻辑验证**：实际运行过 |
| `05-sparse-index-map/sparse_index_map.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `06-ascend-map/ascend_map.py` | 同上 | **CPU 逻辑验证**：实际运行过 |

## 统一说明

* 写作环境见 [`../../../VERSIONS.md`](../../../VERSIONS.md)：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章**没有任何本机实测的性能或正确性结果**。所有需要在加速器上运行的
  示例，预期输出一律写成“正确性判据 + 性能口径”。凡出现数字，只有两类来源：
  **本书模型输出**（可在任意机器复算）与**官方 example / 厂商资料的既有数字**
  （带 corpus 路径与条件）。
* 每个示例目录独立可复制。`04-`/`05-`/`06-` 只有在从本教材仓库内运行时才能找到
  `_vendor/`；找不到时会打印说明并正常退出（不报错）。
