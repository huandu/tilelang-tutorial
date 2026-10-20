# 示例 04：static 与 continuous batching 的调度仿真

## 目的

把“调度决定吞吐”这句话变成一个可以手算复核的离散事件仿真：

1. 同一个请求流、同样多的槽位，static batching 与 continuous batching 分别
   需要多少时间；
2. 输出长度离散（长尾）为什么是 static batching 的杀手；
3. 吞吐、平均延迟、p95 延迟一起看，而不是只看一个数。

## 运行前提

* Python 3.9+，仅标准库（`argparse`、`copy`、`random`、`dataclasses`）。
* **不需要 GPU / NPU，也不需要 TileLang 与 PyTorch。**

## 运行命令

```bash
python3 serving_schedule_sim.py
python3 serving_schedule_sim.py --slots 8 --step-ms 8 --seed 7
```

## 期望输出

默认参数（64 个请求、8 个槽位、decode step 固定 8 ms、
prefill 12 us/token）下的**实际输出**节选：

```text
  输出长度分布          : [16, 24, 32, 48, 64, 96, 128, 256, 512]
  输出长度标准差/均值   : 1.17（离散度越大，static 越吃亏）

[static]
  总耗时 (makespan)      :    23689.5 ms
  吞吐                    :      314.7 token/s
  E2E 延迟 min/mean/p95  :    236.0 /  12084.3 /  22636.6 ms
  TTFT mean              :    11152.3 ms

[continuous]
  总耗时 (makespan)      :     8661.5 ms
  吞吐                    :      860.8 token/s
  E2E 延迟 min/mean/p95  :    219.6 /   3952.8 /   7486.9 ms
  TTFT mean              :     2988.0 ms

对比
  总耗时缩短            : 2.74×
  吞吐提升              : 2.74×
  平均 E2E 延迟         : 12084.3 ms → 3952.8 ms
  p95 E2E 延迟          : 22636.6 ms → 7486.9 ms
```

## 性能口径

**这是仿真输出，不是测量。** 所有参数（`--step-ms`、`--prefill-us-per-tok`、
请求到达间隔、输出长度分布）都是脚本里人为设定的。它证明的是**方向**
（continuous batching 的槽位利用率更高），不是任何具体倍数。

换成别的参数，倍数会变；`--seed` 换一个，分布也会变。要报真实数字，还是要
在真实推理框架上按 `../05-bench-methodology/` 的口径测。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认参数与
若干组 `--slots/--step-ms/--seed`；输出与上面一致。

## 已知局限

* 仿真里 prefill 与 decode **不重叠**（都串在同一条时间轴上）。
  真实的 chunked prefill 会让两者混步，那会让数字更好看，也更复杂。
* decode step 的耗时被假设为常数；真实系统里它随 batch 增大而增大
  （见 `../03-decode-step-share/` 的 batch 敏感度表），
  到某个点再加 batch 就不再摊薄权重流量了。
* 没有模拟抢占、优先级、公平性，也没有模拟 KV 换出。
