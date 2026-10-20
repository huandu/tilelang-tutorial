# 示例 08：用 msProf 采集性能并判读瓶颈（M 树）

## 目的

本章 §8。给一个可改的性能采集骨架：

1. `do_bench(backend="msprof")` 拿延迟（多取几次求中位数）；
2. `do_bench(backend="msprof_detail")` 拿 `KernelProfile`；
3. `read_bottleneck()` 把 AIC/AIV 的 pipe 占比翻译成一句瓶颈结论。

## 关键 API（在 M 树里核对过）

| 项 | 位置 |
| --- | --- |
| `do_bench(..., backend="msprof" | "msprof_detail")` | `tilelang/profiler/bench.py:31-66` |
| 在 NPU 上只允许 `event` / `msprof` / `wall` | `tilelang/profiler/bench.py:199-205` |
| `KernelProfile` 字段（`aic_mad`/`aic_mte*`/`aic_fixpipe`/`aiv_vec`/...） | `tilelang/profiler/msprof.py:60-134` |

## 运行前提

* Ascend NPU + CANN + `torch_npu` + 主干 TileLang `v0.1.15`；
* 环境里能用 msProf（`msprof op` 可执行）；
* `N` 是 `num_blocks * tile = 64 * 8192` 的整数倍。

## 运行命令

```bash
python3 bench_msprof.py --n 1048576 --samples 5
```

## 期望输出

```
correctness passed
latency median = ... us | effective BW = ... GB/s
raw samples (us): [...]
profile: KernelProfile(...)
bottleneck: AIC 最忙的是 ... / AIV 最忙的是 ...
```

**本机没有 NPU，以上输出未产生。** 具体数值必须由你在现场采集；
本书不提供任何昇腾上的实测数字。

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`do_bench` 的 backend 取值、`KernelProfile` 的字段名、
NPU 上允许的 backend 集合逐个在 `_vendor/tilelang-v0.1.15/tilelang/profiler/` 中比对过。
`read_bottleneck()` 的判据来自 `references/profiling.md` 与 `SKILL.md` §12，是本示例
对“怎么读这些比值”的总结，**不是实测结论**。

## 已知局限

* `do_bench(backend="msprof")` 默认会预热并清 cache；跨实现对比时必须保持**同一个**
  cache 策略和 timing 边界。
* `aiv_*` 与 `aic_*` 的比值**各自相对自己的 total_cycles**，不要相加。
* 想区分“一个操作的服务时间”和“它在流水里的等待”，需要 `npusim record` 的指令级
  trace（见 `references/profiling.md` 的 “Operation cycle measurements”），
  本示例不做这件事。
