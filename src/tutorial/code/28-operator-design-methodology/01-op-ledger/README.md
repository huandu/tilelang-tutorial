# 示例 01：算子性能账本

## 目的

把第 28 章 2.1–2.2 节的记账法写成可运行的脚本。给定一个算子和一台机器的
profile，打印：

```
每个 term 的 flops / bytes 明细          （拆到「读 x」「写 y」「rope 表」这一级）
总计 flops、bytes
算术强度 AI = flops / bytes
roof（按 key 选）与它的平衡点 = peak_flops / bandwidth
memory time、compute time、SOL time = max(...)，以及 bound 类型
```

同时打印**如果用另一条 roof 会得到什么结论**——这是本章第 2.3 节「用错 roof」的
可复算版本。

## 运行前提

* Python 3.9+，只用标准库（`argparse`、`dataclasses`、`json`、`math`）。
* **不需要 GPU / NPU / TileLang / PyTorch。**

机器参数来自
`../../../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml`
（标定后）与 `.../05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
（厂商规格），脚本头部的 `MACHINES` 里逐条写了出处。

## 运行命令

```bash
python3 op_ledger.py --case fused-norm-rope-quant
python3 op_ledger.py --case gemm --m 512 --n 512 --k 512
python3 op_ledger.py --sweep
python3 op_ledger.py --machine h100-spec --case gemm --m 4096 --n 4096 --k 4096
python3 op_ledger.py --case rmsnorm --rows 8192 --hidden 4096 --roof cuda_core.fp32
```

## 期望输出

正确性判据：退出码为 0。

`--case fused-norm-rope-quant`（默认参数）的实际输出：

```text
operator : QK-Norm + RoPE + fp8 quant
machine  : NVIDIA H200
shape    : tokens=4096, q_heads=32, kv_heads=8, head_dim=128
TOTAL                                210.207 M        67.765 M

arithmetic intensity :      3.102 FLOP/Byte
roof                 : cuda_core.fp32        57.27 TFLOP/s (CUDA-core FMA)
balance point        :     12.720 FLOP/Byte
memory time          :       15.051 us
compute time         :        3.670 us
SOL time (max)       :       15.051 us   -> memory-bound
```

`--sweep` 的实际输出（节选）：

| 算子 | AI | bound | SOL |
| --- | --- | --- | --- |
| gemm 4096³ fp16 | 1365.33 | compute | 188.463 µs |
| gemm 512³ fp16 | 170.67 | compute | 0.368 µs |
| gemm 128³ fp16 | 42.67 | memory | 0.022 µs |
| attention 32×4096×4096 d128 | 2068.12 | compute | 380.630 µs |
| attention 32×1×8192 d128 | 1.01 | memory | 29.814 µs |
| rmsnorm 8192×4096 | 1.00 | memory | 29.810 µs |
| elementwise chain n=16.8M fused | 0.50 | memory | 14.905 µs |
| elementwise chain n=16.8M eager | 0.25 | memory | 29.810 µs |

其中 rmsnorm 的 29.810 µs、elementwise fused 的 14.905 µs 与第 2 章里用 H200 标定值
手算的结果一致——两份账用的是同一套规则和同一组机器参数。

## 性能口径

**本示例不测量任何性能。** 输出的 `memory time` / `compute time` / `SOL time` 都是
`bytes / bandwidth`、`flops / peak_flops` 的比值，是**下界**。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认、`--sweep`、
`--machine h100-spec` 与多组 shape / roof 参数；输出与本文件一致。没有在 GPU/NPU 上
运行过任何内容。

## 已知局限

* 只统计「输入各读一次、输出各写一次」，中间量一律不计。
* 不考虑 cache 命中、kernel 启动、尾部效应、同步开销。
* 超越函数按约定记 1 FLOP，这不是硬件成本（第 2.5 节、本示例的 `M5`）。
* `--roof` 必须由使用者声明。脚本不会替你判断「这个算子该用哪条 roof」，
  只会把你选的那条和所有其它条一起打印出来作对比。
