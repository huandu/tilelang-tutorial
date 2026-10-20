# 示例 02：随机块序 vs 顺序扫描

## 目的

报告把「块顺序必须随机」写成一条硬性要求
（`.../DeepSelect/docs/DeepSelect-deep-dive.zh.md:43`），并把复杂度证明建立在
「与输入独立的均匀随机块排列」之上（同文件 `:53`）。本脚本用同一条算法在四种输入
分布上、分别用**顺序**与**随机**两种块序跑一遍，把区别摆出来：

* 顺序扫描在大值集中在尾部的输入上会退化成 `A ≈ N`、`W ≈ 1.5N`；
* 随机块序在同样输入上给出 `A` 只有几千、`W` 只有 `N` 的百分之几。

## 运行前提

* Python 3.9+；PyTorch 可选（决定 compact 用 `torch.topk` 还是 `heapq.nlargest`）。

## 运行命令

```bash
python3 block_order_experiment.py
python3 block_order_experiment.py --n 1048576 --k 512
```

## 期望输出

写作机上的**实际输出**（`python3 block_order_experiment.py --n 262144`）：

```
compact step: torch.topk
N=262144  blocks=256  k=512  B=1024  B2=1024
theoretical E[W] bound = 23,517

input      order               A     R           W      W/N  correct
--------------------------------------------------------------------
random     sequential      6,599     5       9,159    0.035  True
random     random          6,597     5       9,157    0.035  True

ascending  sequential    262,144   255     392,704    1.498  True
ascending  random          6,144     5       8,704    0.033  True

descending sequential      2,048     1       2,560    0.010  True
descending random          8,192     7      11,776    0.045  True

staircase  sequential    262,144   255     392,704    1.498  True
staircase  random          6,144     5       8,704    0.033  True
```

其中 `A` 是加入候选缓冲区的元素数、`R` 是循环内 compact 次数、`W = A + k·R`。

## 怎么读这张表

* **随机块序是稳的**：四种输入下 `W/N` 都在 0.03–0.05 之间，和输入分布无关——
  这正是期望复杂度证明要保证的。
* **顺序扫描好不好，取决于大值在哪里**：`descending`（大值在前）时顺序扫描最省
  （`W = 2,560`，比随机还少）；`ascending`/`staircase`（大值在后）时阈值迟迟不涨，
  每个块都被整块接受，`A` 直接等于 `N`。
* 三种形状都**算得对**（最后一列全 `True`）：随机化影响的是**代价**，不是**结果**。

## 验证级别

**CPU 逻辑验证**：实际运行过。**没有在任何加速器上运行过**，表里的时间/数量都是
算法计数，不是 kernel 计数。

## 已知局限

* 这是**算法层**的复现：真实 kernel 里「顺序退化」会表现成带宽被打满、compact 次数
  暴涨，本脚本只数元素个数，不测时间。
* 每组只跑一次随机排列（`--seed` 固定）。要看方差可以多换几个 `--seed`。
