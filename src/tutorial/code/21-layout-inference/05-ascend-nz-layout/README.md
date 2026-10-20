# 05 · 昇腾：分形布局与四个 pass

## 目的

对应第 21 章 §5。演示昇腾侧的两件事：

1. 用 `make_ascend_major_k_layout` / `make_ascend_l0c_layout` 理解
   **分形物理形状**（`C0 = 32 字节 / 元素大小 = 256 / bits`）；
2. 用 `try_extract_fractal_layout` 读出布局的语义标签（kind / c0 / axis），
   理解 `NormalizeAscendFractalStorage` 为什么需要“canonical tag”才能工作
   （`_vendor/tilelang-v0.1.15/src/ascend/layout/ascend_layouts.h:82-120`）。

## 运行前提

* 昇腾 NPU + CANN + TileLang `v0.1.15`（含昇腾后端）。
* `make_ascend_*_layout` 需要一个真实的 `tirx.Buffer`，所以必须在 kernel 作用域里
  构造；本脚本把它放进一个最小 `T.gemm` 里。

## 运行命令

```bash
python3 ascend_nz_layout.py --dry-shape   # 只打印公式推导（纯 Python，不需 NPU）
python3 ascend_nz_layout.py --compile     # 需要 NPU：编译并查看推断布局
```

## 预期输出

`--dry-shape` 打印 C0 与物理形状的**公式计算**（例如 bf16、`[64, 48]` →
`[3, 4, 16, 16]`）。`--compile` 打印编译器为 L0A/L0B/L0C 推断出的布局。

**`--compile` 的输出需要 NPU 才能得到；本机没有 NPU，本章不提供该输出。**

## 验证级别

* `--dry-shape` 分支是纯 Python 公式复算，**CPU 逻辑可复现**（与
  `code/01-layout-algebra` 同级）；C0 公式来自
  `_vendor/tilelang-v0.1.15/src/ascend/layout/ascend_layouts.cc:111-115`。
* `--compile` 分支**仅静态核对**：`make_ascend_major_k_layout` /
  `make_ascend_l0c_layout` / `try_extract_fractal_layout` 的签名在
  `tilelang/layout/ascend.py:20-206` 核对过；**未在硬件上运行**。
