# 02 · 多缓冲的片上占用预算（纯标准库）

## 目的

回答一个写 kernel 时天天要问的问题：**这份 tile 尺寸 + stage 数，放得下吗？**

估算公式（推导见正文第 4.5 节）：

```text
每份 A tile 字节 = block_M × block_K × sizeof(dtype)
每份 B tile 字节 = block_K × block_N × sizeof(dtype)
上界占用        = num_stages × (A + B)
```

这个乘积是**上界**：TileLang 会合并 shared 分配、并复用 live range 不重叠的存储
（`MergeSharedMemoryAllocations`），所以实际占用可能更小。反过来也要注意：累加器
fragment 走寄存器、barrier 还另有开销，所以这个数字不是「剩余全部可用」。

## 运行前提

Python 3.9+，只用标准库。不需要 GPU/NPU，不需要 TileLang。

## 运行命令

```bash
python3 smem_budget.py                                  # 128×128×32，fp16，stages=1..6
python3 smem_budget.py --bm 256 --bn 256 --bk 64 --stages 2 3 4
python3 smem_budget.py --sweep                          # tile 尺寸 × BK 的最大可行 stage 表
python3 smem_budget.py --capacity 32678 --stages 2 3     # 换成很小的预算
python3 smem_budget.py --csv > budget.csv               # 机读输出
```

## 期望输出（本机实际运行结果）

```text
$ python3 smem_budget.py --bm 256 --bn 256 --bk 64 --stages 2 3 4
tile: A[256,64] + B[64,256] @ fp16 (2 B/elem)
per stage (A tile + B tile): 65536 B
budget: 233472 B (228 KiB)
largest stage count that fits: 3

+------------+----------------------------------------------------------+
| num_stages | upper-bound shared footprint                             |
+------------+----------------------------------------------------------+
| 2          |    131072 B    128.0 KiB   fits (56% of budget)          |
| 3          |    196608 B    192.0 KiB   fits (84% of budget)          |
| 4          |    262144 B    256.0 KiB   does NOT fit (112% of budget) |
+------------+----------------------------------------------------------+
```

`--sweep` 会再多打一张表，例如 `block_M x block_N = 256 x 256`、`BK = 64` 时最大
可行 stage 数是 3。

正确性判据：对每一行手工验算 `num_stages × 65536`，并确认 verdict 在
`total > budget` 时翻转。这个工具只做整数乘法与比较，没有别的不确定性。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过，上面引用的输出就是
本机跑出来的。**没有**用 TileLang 编译某个真实 kernel 去对照它的实际 shared
allocation——那需要 GPU。

## 关于 228 KiB 这个数

`h100-sm-shared-228kb` 这个预设来自厂商公开资料里 H100 的
「Shared Memory Size / SM | Configurable up to 228 KB」，指**每个 SM 可配置的 shared
memory 上限**（该 SM 的 shared + L1 合计 256 KB）。它不是「每个 block 一定能申请到
228 KB」的承诺：单 block 能申请多少，还要看 driver/运行时允许的上限与其它资源占用。
把 228 KB 当预算用是保守做法；真要压到极限，请以目标机器上编译+启动的实测结果为准。

出处：`../../references.md` 的 R5 与 R6。
