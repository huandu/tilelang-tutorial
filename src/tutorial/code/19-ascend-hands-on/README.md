# 第 19 章 代码

本章是昇腾动手章，代码分三类：

1. **纯标准库 / 纯探测脚本**（`01-env-check/`、`12-tuning-sweep/`）——不需要 NPU，
   在写作机上**实际运行过**，输出见各自的 README。
2. **适配器仓库（A 树）的昇腾 kernel**（`02`、`03`、`04`、`09`、`10`、`11`）——
   需要真实 Ascend NPU + CANN + torch_npu + `_vendor/tilelang-ascend-v0.1.2.000`。
   本章对这些示例只做到**仅静态核对**：`python3 -m py_compile` 过语法、逐个在
   `_vendor/` 里比对 API 名与签名。**没有在硬件上运行过任何一条。**
3. **主干（M 树）的昇腾 kernel**（`05`、`07`、`08`）——需要真实 Ascend NPU +
   `_vendor/tilelang-v0.1.15`（`tilelang.ascend.language`）。同样只做到**仅静态核对**。
   `06-ascendc-baseline/` 是一段 Ascend C 参考代码，**不参与 Python 语法检查**。

## 目录

```
code/
├─ README.md                       本文件
├─ 01-env-check/                   纯标准库；探测 CANN / torch_npu / TileLang 环境
├─ 02-elementwise-ub/              A 树；最小逐元素 add（GM→UB→V→GM）
├─ 03-elementwise-pipeline/        A 树；手写双缓冲的逐元素 add
├─ 04-gemm-l1-l0c/                 A 树；最小 GEMM（GM→L1→L0C→GM）
├─ 05-gemm-dialect/                M 树；GEMM+ReLU（dual_copy + SimtVF 混核）
├─ 06-ascendc-baseline/            Ascend C；同一逐元素算子的手写写法（C++ 参考）
├─ 07-vector-modes/                M 树；SIMT / SIMD 两种向量写法 + buffer 版本注解
├─ 08-msprof-bench/                M 树；do_bench(backend="msprof") 采集与瓶颈判读
├─ 09-aclgraph/                    A 树；ACLGraph capture + replay
├─ 10-debug-and-dump/              A 树；T.printf / T.dump_tensor
├─ 11-exception-dump/              A 树；AICore exception dump
└─ 12-tuning-sweep/                纯标准库；配置扫描驱动（先定测什么，再上机器）
```

每个子目录都能独立复制出去运行，不依赖其他章节或其他示例的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-env-check/env_check.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 |
| `02-elementwise-ub/vec_add_adapter.py` | NPU + CANN + torch_npu + A 树 | **仅静态核对**；未运行 |
| `03-elementwise-pipeline/vec_add_pipeline.py` | 同上 | **仅静态核对**；未运行 |
| `04-gemm-l1-l0c/gemm_adapter.py` | 同上 | **仅静态核对**；未运行 |
| `05-gemm-dialect/gemm_relu_dialect.py` | NPU + CANN + torch_npu + M 树 | **仅静态核对**；未运行 |
| `06-ascendc-baseline/add_ascendc.cpp` | NPU + CANN + Ascend C 编译环境 | **仅静态核对**（C++，未编译） |
| `07-vector-modes/vecadd_modes.py` | NPU + CANN + torch_npu + M 树 | **仅静态核对**；未运行 |
| `08-msprof-bench/bench_msprof.py` | NPU + CANN + torch_npu + M 树 + msProf | **仅静态核对**；未运行 |
| `09-aclgraph/rms_rope_aclgraph.py` | NPU + CANN + torch_npu + A 树 | **仅静态核对**；未运行 |
| `10-debug-and-dump/print_and_dump.py` | NPU + CANN + torch_npu + A 树 | **仅静态核对**；未运行 |
| `11-exception-dump/exception_dump.py` | NPU + CANN ≥ 9.3.0 + A 树 | **仅静态核对**；未运行 |
| `12-tuning-sweep/tune_sweep.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过 |

## 统一说明

* 写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11，
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此任何需要加速器才能得到的数字（延迟、吞吐、算力利用率）在 `code/` 里都不存在；
  性能对比的口径与来源见 `../references.md`。
* 版本一律为 TileLang `v0.1.15`（M 树）与 TileLang-Ascend `v0.1.2.000-release`（A 树）；
  本章只引用这两个锁定版本，不引用未锁定的开发分支。
* `02`–`05`、`07`–`11` 这些示例里的 `import` 语句都按各自所属的树写死了。**不要**把
  两个包里的示例混在同一个进程里跑。
