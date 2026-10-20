# 示例 07：SIMT / SIMD 两种向量写法 + buffer 版本注解（M 树）

## 目的

本章 §7.4–§7.5。用一个 N=2²⁰ 的向量加，对照 M 树的两条 Vector 编程路径，
并演示 `T.annotate_buffer_versions` 如何让编译器代管三级多缓冲。

| 模式 | 写法 | 线程概念 | 典型场景 |
| --- | --- | --- | --- |
| `simt` | `with T.SimtVF(threads=N): for i in T.Parallel(tile): ...` | 有，编译器映射 | 归约、归一化、条件逻辑 |
| `simd` | `with T.SimdVF(): T.simd.vld/vadd/vsts(mask)` | 无，2048-bit 向量 | 规整、可掩码的逐元素运算 |

来源：`_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py`。

## 运行前提

* Ascend 950 NPU + CANN + `torch_npu` + 主干 TileLang `v0.1.15`；
* `N` 必须是 `tile * num_blocks = 8192 * 64 = 524288` 的正整数倍。

## 运行命令

```bash
python3 vecadd_modes.py --n 1048576                    # 正确性（simt + simd）
python3 vecadd_modes.py --mode simd --n 1048576         # 只跑 simd
python3 vecadd_modes.py --bench --n 1073741824          # 顺带用 msprof 测带宽
```

## 期望输出

```
vecadd (simt): correctness passed
vecadd (simd): correctness passed
```

`--bench` 时额外打印 `... us | ... GB/s`（数值来自你机器上的 msProf）。

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`T.Kernel`（无 `threads=`）、`T.alloc_shared`、
`T.annotate_buffer_versions`、`T.Pipelined`、`T.SimtVF`、`T.SimdVF`、`T.simd.pset/vld/vadd/vsts`、
`do_bench(backend="msprof")` 逐个在 `_vendor/tilelang-v0.1.15/` 中比对过。

## 已知局限

* 带宽口径是「两次输入读 + 一次输出写」，与官方示例一致；对比不同实现时**口径必须一致**。
* `--bench` 用的 `N=2³⁰` 会占用约 12 GB 显存（3 × 2³⁰ × 4B），小卡请调小 `--n`。
