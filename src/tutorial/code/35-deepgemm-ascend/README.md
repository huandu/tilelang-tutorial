# 第 35 章 代码

本章的代码分两类：

1. **能在写作机上跑出结论的**——`02-perf-ledger/` 与 `03-config-search/`。
   它们把 `DeepGEMM-Ascend` 源码里的硬件模型、容量约束与 README 的性能表
   用纯 Python 复算一遍，**只用标准库**，不需要 numpy、GPU、NPU。
2. **只能静态阅读的**——`01-python-api/` 与 `04-hc-prenorm-tilelang/`。
   前者需要昇腾 NPU + CANN + `torch_npu` + 已安装的 `deep_gemm`；
   后者需要昇腾 NPU + TileLang（`tilelang.ascend.language`）。
   本章对它们只做到**仅静态核对**：语法用 `py_compile` 过、API 名与签名逐个在
   corpus 快照里比对过。**没有在硬件上运行过任何一条。**

四个示例互相独立，可以分别复制出去运行。

## 目录

```
code/
├─ README.md                              本文件
├─ 01-python-api/deepgemm_ascend_call.py  照 tests/ 写的最小调用样例（BF16 / FP8 / M-grouped）
├─ 02-perf-ledger/perf_ledger.py          纯标准库；把 README 的性能表换算成利用率与带宽，并复算硬件极限
├─ 03-config-search/config_search.py      纯标准库；从 heuristics.hpp 提炼的候选/容量模型教学版
└─ 04-hc-prenorm-tilelang/hc_prenorm_sketch.py  从 hc_prenorm.py 提炼的 TileLang 骨架
```

## 验证级别总表

| 示例 | 它交付什么 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- | --- |
| `01-python-api/deepgemm_ascend_call.py` | DeepGEMM-Ascend 的三类最小调用（BF16 dense、FP8 dense、M-grouped），含 SF 打包的示例造法 | 昇腾 NPU + CANN 9.20 + `torch_npu` + `deep_gemm` | **仅静态核对**：`py_compile` 通过；函数名与参数逐个对照 `csrc/apis/gemm.hpp` 的注册表；**未运行** |
| `02-perf-ledger/perf_ledger.py` | 用源码里的硬件模型复算 bf16/fp8/fp4 峰值；把六张表的 Computation 换算成利用率；用“A+B(+SF)”口径复算带宽并与表格列对比 | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0，输出见该目录 README |
| `03-config-search/config_search.py` | 用 `heuristics.hpp` 的容量公式枚举 `block_m/block_n` 与 `mad_m/n/k`，打印哪些候选合法、L1/L0 各能放几级 | Python 3.9+（只用标准库） | **CPU 逻辑验证**：实跑过，退出码 0，输出见该目录 README |
| `04-hc-prenorm-tilelang/hc_prenorm_sketch.py` | 把 `hc_prenorm.py` 的骨架（`get_config` + `MixedKernel` + SIMD 阶段 + K-split 原子加）提炼成可读版本，逐段加注释 | 昇腾 NPU + TileLang（M 树 `tilelang.ascend.language`） | **仅静态核对**：`py_compile` 通过；`T.MixedKernel`/`T.alloc_l1`/`T.dual_copy`/`T.SimdVF`/`T.PerCoreTask`/`T.set_atomic` 等逐个对照 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/`；**未运行** |

## 统一说明

* 写作环境见 [`../../VERSIONS.md`](../../VERSIONS.md)：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此 `01`、`04` 的所有结论都是**静态阅读**的结果；`02`、`03` 的所有数字都是
  **本机 CPU 运算**的结果（复算，不是测量）。
* 版本：TileLang `v0.1.15`（`_vendor/tilelang-v0.1.15/`）；DeepGEMM-Ascend 取自
  corpus 快照 commit `8491bbb4b8c0`（见 corpus 的 `PROVENANCE.md`）。
* `02` 引用的“硬件极限”复算公式来自 `csrc/jit_kernels/heuristics.hpp:46-48`；
  它是**模型**，不是任何一台机器的实测值。

## 建议的阅读顺序

1. 读完正文第 5 节后跑 `02-perf-ledger/`：先把“98%–99.8%”换算成你自己能量出来的量；
2. 读完第 4.4 节后跑 `03-config-search/`：亲手确认“为什么 `256×256` 已经用满 L0C”；
3. 读完第 3 节后看 `01-python-api/`：把调用姿势与注册表对上；
4. 读完第 6 节后看 `04-hc-prenorm-tilelang/`：把骨架与 `hc_prenorm.py` 的行号对上。

