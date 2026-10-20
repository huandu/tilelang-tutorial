# 示例 05：推理基准的口径换算与污染检查

## 目的

把“报一个推理性能数字之前必须做的事”变成一份可打印的检查表：

1. TTFT / TPOT / ITL / E2E / 吞吐 / 有效吞吐的定义与关系；
2. 同一组数用三种 TPOT 算法算出来的差异；
3. 首次调用（JIT 编译、autotune、显存池首次增长、graph capture）落进
   测量窗口会把均值抬高多少；
4. 四类“看起来是真的”的假数字：KV/prefix cache、L2 命中、图模式混测、
   样本太少；
5. 报告前必须写下来的 12 项条件。

## 运行前提

* Python 3.9+，仅标准库（`argparse`）。
* **不需要 GPU / NPU，也不需要 TileLang 与 PyTorch。**

## 运行命令

```bash
python3 bench_methodology.py
python3 bench_methodology.py --ttft 420 --e2e 9000 --out-tokens 512
```

## 期望输出

默认参数下的**实际输出**节选：

```text
  TTFT = 420.00 ms，E2E = 9,000.00 ms

  TPOT = (E2E − TTFT) / (n − 1) = 16.79 ms   ← 推荐
  TPOT = (E2E − TTFT) / n       = 16.76 ms   ← 少数派口径
  E2E / n                       = 17.58 ms   ← 把 TTFT 摊进每 token，不能叫 TPOT

  三种算法的差异：0.20% 与 4.90%

  稳态单次耗时           : 0.350 ms
  首次耗时（含编译/autotune）: 3,200.0 ms
  测量 100 次
  → 含首次：3,235.0 ms（均值 32.350 ms）
  → 去掉首次：35.0 ms（均值 0.350 ms）
  → 均值被抬高 92.43×

  ● KV / prefix cache 命中 / L2 cache 命中 / 图模式混测 / 样本太少
    （每类都给出症状、原因、做法）

  报告一个推理性能数字前，必须写下来的清单：12 条
```

## 性能口径

**本示例不测量任何性能。** 里面的 3200 ms“首次耗时”与 0.350 ms
“稳态耗时”是演示用的假设值，就是为了说明污染效应有多大。

脚本引用的真实口径只有两处，都在锁定版本源码里：

* `do_bench` 的默认 `warmup=25` ms、`rep=100` ms
  （`_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31-34`）；
* `do_bench` 默认 flush 256 MB L2（同文件 `cache_size: int = 256`）。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认参数与
自定义 `--ttft/--e2e/--out-tokens`；输出与上面一致。

## 已知局限

* 不判断你的对比是否公平；它只是把“应该写下来的东西”列出来。
* 没有覆盖分布式场景的口径问题（TP/EP 通信占比、跨节点带宽、
  不同 rank 的尾延迟）。
* TTFT 的起点有“请求到达”“完成 scheduler 入队”“prefill 开始”三种定义，
  脚本只提醒你要写清，不替你选。真正测量时的工具用法见第 12 章。
