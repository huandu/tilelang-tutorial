# 第 05 章 代码

本章是**对比**章，代码的作用不是“跑出性能”，而是把同一个算子在不同抽象层次上的
写法并排摆出来，让你看见“少掉的代码被谁吃掉了”。

## 目录

| 目录 | 内容 | 是否需要加速器 |
| --- | --- | --- |
| `01-unified-gemm-relu/` | 统一例子 `C = relu(A @ B)` 在六个系统里的写法（含伪代码与真实片段） | 见各文件说明 |
| `02-abstraction-layers/` | 一个**纯 Python、可在 CPU 上运行**的脚本，打印本章的抽象层次表与决策表 | 不需要 |

## 验证级别说明（与 WORK-PLAN §3.4 一致）

写作环境为 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章：

* `01-unified-gemm-relu/` 下的所有文件都是 **`仅静态核对`**：
  * 涉及 TileLang API 的，逐个在 `_vendor/tilelang-v0.1.15/` 中比对了名字与签名；
  * 涉及 Triton / CUDA / CUTLASS-CuTe / Ascend C / PTO-DSL 的，标注了来源，
    并明确区分“真实片段”与“教学伪代码”；
  * **没有任何一个文件在本机执行过**。
* `02-abstraction-layers/abstraction_layers.py` 是唯一可以真正跑起来的脚本，
  其验证级别是 **`在 CPU 上验证过逻辑`**（纯 Python，无第三方依赖）。

## 运行

```bash
# 只跑得起来的那个（CPU）
python3 02-abstraction-layers/abstraction_layers.py

# 语法静态检查（不 import、不需要硬件）
python3 -m py_compile 01-unified-gemm-relu/*.py 02-abstraction-layers/*.py
```

在真实硬件上如何验证各片段，见 `01-unified-gemm-relu/README.md` 末尾的“在有卡机器上
怎么验证”。
