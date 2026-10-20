# 第 26 章代码目录

本章的六个示例按「能不能在没有加速器的机器上跑」分成两组。

## A 组：不需要 GPU / NPU，也不需要安装 tilelang

| 目录 | 做什么 | 验证级别 |
| --- | --- | --- |
| `01-runtime-anchor-check/` | 两层核对：172 条锚点表 + 全章 241 处行内引用的存在性/范围/非空校验 | **在本机运行过**（2026-10-01），`172/172` 通过、扫描 0 问题 |
| `02-cache-dir-inspect/` | 读取真实缓存目录并复刻 `KernelCache` 的 manifest 校验；`--selftest` 用自造 fixture 验证读取逻辑 | **在本机运行过**（2026-10-01），selftest 13 项判据全过 |
| `03-packed-abi-sim/` | 用纯 Python 复刻 `MakePackedAPI` / `tvm_ffi_binder` 的参数检查规则与报错文案 | **在本机运行过**（2026-10-01），13 个场景全部符合源码描述 |

A 组三个示例都只用标准库：本机是 macOS（Apple Silicon），没有 NVIDIA GPU、
没有昇腾 NPU，也没有安装 `tilelang` / `torch`；A 组的价值恰恰在于它在这样的机器上
仍然可重跑。

## B 组：需要真实硬件，本章只做静态核对

| 目录 | 需要什么 | 验证级别 |
| --- | --- | --- |
| `04-torch-stream-call/` | CUDA GPU + `tilelang` + `torch` | **仅静态核对**：`python3 -m py_compile` 通过，API 名逐个与 `_vendor/tilelang-v0.1.15/` 比对；**未运行** |
| `05-compile-only-inspect/` | 任意装得上 `tilelang==0.1.15` 的 CPU 机器 | **仅静态核对**：同上；**未运行** |
| `06-ascend-aclgraph/` | 昇腾 950 + CANN + `torch_npu` | **仅静态核对**：`python3 -m py_compile` 通过，与 `_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/` 的官方示例逐行比对；**未运行** |

B 组的每个目录在自己的 `README.md` 里写了运行前提、运行命令、期望输出与正确性判据，
方便你在有卡机器上把它变成真正的验证。

## 快速开始

```bash
cd chapters/26-runtime-jit-abi/code
python3 01-runtime-anchor-check/verify_runtime_anchors.py --quiet
python3 02-cache-dir-inspect/inspect_cache_dir.py --selftest
python3 03-packed-abi-sim/packed_abi_sim.py
```

三条命令在我们的写作环境里都返回退出码 0。

## 关于这些示例不做的事

没有一个示例宣称跑出了性能数字。本章唯一的量化数据引用自未合入 PR 的原始表格
（见 `references.md` 的 R4），并标注了「仅 main / 未合入」与测量条件。
