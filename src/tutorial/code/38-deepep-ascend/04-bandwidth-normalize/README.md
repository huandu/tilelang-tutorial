# 示例 04：把 README 性能表换算成“相对物理上限”

## 目的

1. 原样打印 `DeepEP-Ascend/README.md:13-19` 的性能表（区间上下限）；
2. 反推 `README.md:21` 那句话（“Sustained dispatch reaches roughly 90–95%
   of the physical payload bandwidth limit for EP sizes up to 32”）
   隐含的物理上限量级。

## 运行前提

Python 3.9+，仅标准库。

## 运行命令

```bash
python3 bandwidth_normalize.py
python3 bandwidth_normalize.py --low 0.9 --high 0.95
```

## 期望输出

```text
原始表（DeepEP-Ascend/README.md:13-19，单位 GB/s，区间覆盖所有 rank）
  EP       Dispatch        Combine   combine/dispatch
   8    373-375       345-347                92.5%
  16    348-352       338-341                97.0%
  32    335-340       320-324                95.4%
  64    323-327       294-298                91.1%
 128    313-320       272-278                86.9%

相对 EP8 的下降（取区间中点）
  EP16   dispatch 比 EP8 低  6.4%
  EP32   dispatch 比 EP8 低  9.8%
  EP64   dispatch 比 EP8 低 13.1%
  EP128  dispatch 比 EP8 低 15.4%

反推物理 payload 带宽上限（dispatch 数字 ÷ 0.90…0.95）
  EP8    373-375 GB/s  →  上限 393–417 GB/s
  EP16   348-352 GB/s  →  上限 366–391 GB/s
  EP32   335-340 GB/s  →  上限 353–378 GB/s
  三者并集 ≈ 353–417 GB/s/rank
```

## 三个必读的提醒

1. 表里的 GB/s 是**有效 payload 字节 / 时间**，口径见
   `tests/ep/test_ep.py:404-406`（不含 padding、不含 barrier）。
2. 计时含 issue 与 drain，**不含最后的 epilogue**（`README.md:21`）。
3. README 原文是 "roughly 90–95%"，所以这只是**区间估计**；
   corpus 里没有 Ascend 950 UBMEM 链路的官方带宽规格，
   **不能当成硬件手册上的数字**（见 `../references.md` 的“待补资料”）。

## 验证级别

**CPU 逻辑验证**：作者在 macOS (Apple Silicon) / Python 3.11 上实际运行过。
输入数字逐字来自 README；反推结果明确标注为区间估计，**不是实测**。
