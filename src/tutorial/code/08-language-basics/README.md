# 第 08 章 代码

本章讲语言基础，代码的使命是把“语言表面的写法”和“编译器实际做的事”对上：

1. `01-kernel-grid/` — `T.Kernel` 的网格语义：位置参数怎么对应 x/y/z、
   `threads=` 是什么、块内怎么取索引。附一个纯 Python 的网格模型。
2. `02-dtypes/` — dtype 家族清单与 fp8/fp6/fp4 的写法；附一个把
   `_vendor/tilelang-v0.1.15/tilelang/language/dtypes.py` 的 `_all_dtypes`
   逐条比对回去的脚本。
3. `03-memory-scopes/` — `alloc_shared / fragment / local / var` 的语义与选择依据，
   以及昇腾侧 `alloc_l1 / l0a / l0b / l0c / UB` 的拼法差异；附一个占用计算器。
4. `04-loops/` — `T.serial / vectorized / unroll / Parallel / Pipelined` 的语义，
   以及 `T.Parallel` 的维度映射规则；附一个把映射公式复现出来的脚本。
5. `05-control-flow/` — `if/else`、`T.if_then_else`、`T.Select`、`while`，
   以及控制流对向量化与流水的影响。
6. `06-python-boundary/` — Python 语法在 kernel 里的可用边界；附一个教学用的
   静态扫描器。

每个示例目录都能单独复制出去，不依赖其他章节的文件。

## 目录

```text
code/
├─ README.md                          本文件
├─ 01-kernel-grid/
│  ├─ README.md
│  ├─ kernel_grid.py                  TileLang 示例；仅静态核对
│  └─ grid_index_model.py             纯标准库；CPU 逻辑验证
├─ 02-dtypes/
│  ├─ README.md
│  ├─ dtype_families.py               纯标准库；CPU 逻辑验证（含 --diff）
│  └─ fp8_fp4_exprs.py                TileLang 示例；仅静态核对
├─ 03-memory-scopes/
│  ├─ README.md
│  ├─ scope_footprint.py              纯标准库；CPU 逻辑验证
│  └─ alloc_scopes.py                 TileLang 示例；仅静态核对
├─ 04-loops/
│  ├─ README.md
│  ├─ loop_mapping.py                 纯标准库；CPU 逻辑验证
│  └─ loops_demo.py                   TileLang 示例；仅静态核对
├─ 05-control-flow/
│  ├─ README.md
│  └─ control_flow_demo.py            TileLang 示例；仅静态核对
└─ 06-python-boundary/
   ├─ README.md
   └─ py_compat_scan.py               纯标准库；CPU 逻辑验证（含 --self-test）
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-kernel-grid/grid_index_model.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上实际运行过，输出见该目录 README |
| `01-kernel-grid/kernel_grid.py` | TileLang v0.1.15 + NVIDIA GPU | **仅静态核对**：`python3 -m py_compile` 通过，API 逐个在 `_vendor/` 里核对；**未运行** |
| `02-dtypes/dtype_families.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：实际运行过；`--diff` 与 vendor 的 `_all_dtypes` **完全一致**（173 条） |
| `02-dtypes/fp8_fp4_exprs.py` | TileLang v0.1.15（部分函数需 GPU） | **仅静态核对**：`py_compile` 通过，dtype 符号与 helper 逐个核对；**未运行** |
| `03-memory-scopes/scope_footprint.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：实际运行过，输出见该目录 README |
| `03-memory-scopes/alloc_scopes.py` | TileLang v0.1.15（昇腾部分需 NPU） | **仅静态核对**：`py_compile` 通过，`alloc_*` 签名逐个核对；**未运行** |
| `04-loops/loop_mapping.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：实际运行过，输出见该目录 README |
| `04-loops/loops_demo.py` | TileLang v0.1.15 + NVIDIA GPU | **仅静态核对**：`py_compile` 通过，循环构造子签名逐个核对；**未运行** |
| `05-control-flow/control_flow_demo.py` | TileLang v0.1.15 + NVIDIA GPU | **仅静态核对**：`py_compile` 通过，`if_then_else`/`Select`/`while` 逐个核对；**未运行** |
| `06-python-boundary/py_compat_scan.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：实际运行过，5/5 self-test 通过，输出见该目录 README |

## 统一说明

* 本章写作环境（见 `../../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11
  （miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章没有任何本机实测的运行结果或性能数字。凡是写着“仅静态核对”的示例，
  作者做到的只是“语法能被 `py_compile` 接受 + 每个 API 名与签名在
  `_vendor/tilelang-v0.1.15/` 里找得到”；**这不等于跑通**。
* `scope_footprint.py`、`grid_index_model.py`、`loop_mapping.py` 是**模型**，
  不是硬件测量；它们把公式和假设写清楚，以便被复算与证伪。
* 版本一律为 TileLang `v0.1.15`；本章不使用任何含糊的版本说法。昇腾适配器仓库
  单独标注为 `_vendor/tilelang-ascend-v0.1.2.000`。
* 本章所有 `.md` / `.py` 文件都不含遗留标记（`check_chapter.py` 会扫这一项）。
