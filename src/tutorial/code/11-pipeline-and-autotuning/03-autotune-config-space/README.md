# 03 · 搜索空间的生成、剪枝与排序（纯标准库）

## 目的

自动调优最常见的失败不是“没调出最优解”，而是**搜索空间本身就是错的**：

* 全部候选都不可行（编译/启动就失败），白烧一晚上；
* 只有一个候选可行，调优等于没调；
* 候选很多，但代价模型分不出差别，排名是噪声。

这个脚本把上面三件事变成可复现的输出。它是
`_vendor/tilelang-v0.1.15/tilelang/autotuner/` 的**离线替身**：只复现“决定这次搜索
值不值得跑”的那部分逻辑，不涉及编译与测量。

## 运行前提

Python 3.9+，只用标准库。不需要 GPU/NPU，不需要 TileLang。

## 运行命令

```bash
python3 config_space.py                  # 一个形状合理的空间
python3 config_space.py --mode all       # 三种空间一次跑完
python3 config_space.py --mode all-infeasible
python3 config_space.py --mode degenerate --topk 5
```

## 期望输出（本机实际运行结果，`--mode all` 摘录）

```text
mode = well-formed
problem: M=4096 N=4096 K=4096 fp16, shared budget 228 KiB
configurations requested: 216
configurations legal:     214
distinct cost values:     12
  rejected (shared memory over budget): 2
...
mode = all-infeasible
configurations requested: 48
configurations legal:     0
=> This space is wasted: every candidate is infeasible.

mode = degenerate
configurations requested: 1
configurations legal:     1
distinct cost values:     1
=> Only one legal configuration: tuning cannot rank anything.
```

正确性判据（都可手工核对）：

1. `well-formed` 的 216 个候选里，被拒的 2 个恰好是 `block_M = block_N = 256,
   block_K = 64, num_stages = 4`（4 × 65536 B = 256 KiB > 228 KiB）的两个线程数变体；
2. `all-infeasible` 的 `legal` 必须为 0；
3. `degenerate` 的 `legal` 与 `distinct cost values` 都必须为 1。

## 关于代价模型

`cost()` 是一个**故意粗糙的解析代理**，只用到本章两个概念：

* 每一轮 reduction 的步进成本 = `max(copy_issue, compute)`；
* copy engine 同时只能在途 `num_stages` 份搬运，所以按 Little 定律，
  持续步进成本是 `max(copy_issue, latency / num_stages)`，直到深度足够把延迟盖住。

它不建模寄存器压力、occupancy、wave quantization、bank conflict。所以：

* 它可以用来看“哪些配置注定不行”和“哪些配置明显更差”；
* 它**不能**用来宣布某个配置就是最优——那要靠真机测量。

输出里的 `distinct cost values` 就是这个问题的一个体检指标：如果它很小，说明模型在
这个空间上已经用完了分辨率，排名不可信。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过，上面引用的输出就是
本机跑出来的。**没有**用真实 GPU 跑过对应的 kernel。
