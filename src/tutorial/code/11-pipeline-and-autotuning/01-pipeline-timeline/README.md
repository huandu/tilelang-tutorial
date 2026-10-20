# 01 · 流水时空模型（纯标准库）

## 目的

把“不做流水会怎样”“多深才够”变成可以自己复算的算术，而不是靠感觉。它模拟一个
`copy -> compute` 的 reduction 循环：每个 iteration 有一次搬运（占用 copy engine
`issue` 个周期，数据在 `latency` 个周期后可用）和一次计算（占用 compute 单元
`compute` 个周期）。`stages` 就是 `T.Pipelined(num_stages=...)` 的值，也就是
缓冲区的份数。

调度器是一个确定性的贪心 list scheduler：只要 copy engine 与目标槽位都空闲就发搬运，
只要数据到了且计算单元空闲就开始算。**这个模型不包含任何硬件测量**。

## 运行前提

Python 3.9+，只用标准库。不需要 GPU/NPU，不需要 TileLang。

## 运行命令

```bash
python3 pipeline_timeline.py                       # 默认：stages = 1,2,3,4
python3 pipeline_timeline.py --sweep               # stages = 1..8，看收益何时饱和
python3 pipeline_timeline.py --latency 200 --compute 20   # 延迟更难掩藏的情形
python3 pipeline_timeline.py --stages 1 2 3 --iterations 32
```

参数含义：`--issue` 搬运占用 copy engine 的周期数；`--latency` 数据从发出到可用的
延迟；`--compute` 一次计算的周期数；`--iterations` reduction 循环的轮数。

## 期望输出（本机实际运行结果）

默认参数下：

```text
model: copy -> compute, greedy list schedule, abstract cycles
  iterations=8 issue=1 latency=30 compute=20
  latency/compute = 1.50 -> smallest depth that reaches the steady state = 3

== summary ==
+--------+--------------------------------------------------------------------+
| stages | makespan  speedup  issue_iv  compute_iv                            |
+--------+--------------------------------------------------------------------+
| 1      |   400  x1.00  issue_iv= 50.0  compute_iv= 50.0  compute_util=  43% |
| 2      |   220  x1.82  issue_iv= 24.3  compute_iv= 24.3  compute_util=  84% |
| 3      |   190  x2.11  issue_iv= 18.6  compute_iv= 20.0  compute_util= 100% |
| 4      |   190  x2.11  issue_iv= 15.7  compute_iv= 20.0  compute_util= 100% |
+--------+--------------------------------------------------------------------+
```

正确性判据（不是性能判据）：

1. `stages=1` 时 `compute_iv` 应等于 `latency + compute`（30 + 20 = 50）；
2. `stages` 增大到等于 `1 + ceil(latency / compute)` 之后 `compute_util` 应达到 100%，
   再增大 `stages` 总时长不再变化（本例是 3）；
3. `--sweep` 下 `makespan` 曲线单调不增，并在一处之后变成常数。

渐近（稳态）步长满足

```text
每轮周期 = max( compute, issue, (latency + compute) / depth )
```

第 1 条就是这个式子在 `depth = 1` 时的取值；第 2 条是它从“缓冲区不够”转到
“计算本身是瓶颈”的拐点。注意打印出来的是有限轮数的平均值，会带上填充段的影响，
所以 `stages=2` 那行的 `compute_iv` 是 24.3 而不是式子的 25.0——把 `--iterations`
开大会向 25 收敛。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过，上面引用的输出就是
本机跑出来的。**没有**在任何 GPU/NPU 上对照过——本机没有卡。

## 这个模型不做什么

* 它没有建模 copy engine 的并发上限、bank conflict、wave quantization、寄存器压力；
* 它假设每轮的计算量相同、没有尾部不整除的迭代；
* 它给出的所有数字都是抽象周期数，不能与任何真实 kernel 的微秒数互换。

它的用途是解释图 1 与第 1 节的对比，而不是预测性能。
