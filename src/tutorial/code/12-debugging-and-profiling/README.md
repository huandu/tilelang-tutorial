# 第 12 章 代码

本章是“工具体检章”，所以代码的任务不是实现某个算子，而是**把每个调试/分析工具的
入口与用法固定下来**，并明确区分“本机能做到的”和“必须上卡才能做的”。

六个示例互相独立，可以分别复制出去运行。

## 目录

```
code/
├─ README.md                          本文件
├─ 01-tool-inventory/
│  ├─ README.md
│  └─ tool_inventory.py               纯标准库；在无 GPU/NPU 的机器上可运行
├─ 02-tolerance-model/
│  ├─ README.md
│  └─ tolerance_model.py              标准库 + numpy；可在 CPU 上运行
├─ 03-compile-only/
│  ├─ README.md
│  └─ add_kernel_compile_only.py      需要 tilelang；本机只有 --static-only 路径
├─ 04-lower-trace/
│  ├─ README.md
│  └─ gemm_lower_trace.py             需要 tilelang；本机只有 --static-only 路径
├─ 05-print-and-assert/
│  ├─ README.md
│  └─ debug_print_demo.py             需要 NVIDIA GPU；本机只有 --static-only 路径
└─ 06-ascend-dump/
   ├─ README.md
   └─ ascend_dump_demo.py             需要昇腾 NPU；本机只有 --static-only 路径
```

## 验证级别总表

| 示例 | 它交付什么 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- | --- |
| `01-tool-inventory/tool_inventory.py` | 本章提到的 36 个工具/API 入口，逐个在锁定版本源码里确认存在并打印 `文件:行号` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0，输出见该目录 README |
| `02-tolerance-model/tolerance_model.py` | fp16/bf16 的机器精度、长度 K 的归约误差预算，以及与 fp64 参考的实测误差分布 | Python 3.9+ 与 numpy | **CPU 逻辑验证**：同上实际运行过，退出码 0，输出见该目录 README |
| `03-compile-only/add_kernel_compile_only.py` | 不跑 GPU 就把 kernel 降到源码；`compile_kernel_source()` 与 CLI 两条路径 | 完整可运行需 Linux wheel 的 tilelang；`--static-only` 任意机器 | **仅静态核对**：`py_compile` 通过；5 个入口逐个在 `_vendor/` 里核对；**未安装 tilelang，未生成源码** |
| `04-lower-trace/gemm_lower_trace.py` | `TL_LOWER_TRACE` 全流程、`lower_trace()` 程序 API、legacy `pass_diff` 三种看法 | 完整可运行需 tilelang（CUDA codegen）；`--static-only` 任意机器 | **仅静态核对**：`py_compile` 通过；5 个入口逐个核对；**未生成过任何报告** |
| `05-print-and-assert/debug_print_demo.py` | `T.print` 的四种用法与单线程门控语义、`T.device_assert` | 完整可运行需 NVIDIA GPU + CUDA toolkit；`--static-only` 任意机器 | **仅静态核对**：`py_compile` 通过；4 个入口逐个核对；**未运行 kernel** |
| `06-ascend-dump/ascend_dump_demo.py` | 昇腾侧 `T.printf` / `T.dump_tensor` 与 AICore 异常 dump 的完整调用骨架 | 完整可运行需昇腾 NPU + CANN >= 9.3.0；`--static-only` 任意机器 | **仅静态核对**：`py_compile` 通过；5 个入口逐个核对；**未运行 kernel** |

## 统一说明

* 写作环境见 [../../VERSIONS.md](../../VERSIONS.md)：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章**不包含任何自测性能数字**。正文里出现的性能/容量数字（例如“每个核
  dump 上限 1 MB”）都来自 corpus 的官方文档，并在 `references.md` 里给出出处。
* 需要加速器的四个示例都提供 `--static-only`：在没有卡（或没有装 tilelang）的机器上
  打印入口核对清单并正常退出（退出码 0）；不带该参数运行时会尝试导入 `tilelang`
  或 `torch`，缺少依赖时会以非 0 退出。
* 版本一律为 TileLang `v0.1.15`
  （commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）与 TileLang-Ascend
  `TileLang-ascend-v0.1.2.000-release`
  （commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。

## 建议的阅读顺序

1. 先跑 `01-tool-inventory`，拿到本章那张工具表的机器输出；
2. 再跑 `02-tolerance-model`，理解“容差从哪来”；
3. 然后按正文 3.1 → 3.6 的顺序读 `03`～`06`，对照正文理解每个工具的时序与限制。
