# 示例 1：朴素 attention 与分块融合 attention 的访存记账

## 目的

正文第 1.2 节说「朴素实现要在 HBM 上读写四趟 `Sq×Skv` 的中间结果」，第 2.5 节说
「flash attention 把它们留在片上」。这个示例把那两句话变成**可以逐项对数**的整数记账：

* `naive`：`S = QKᵀ`、`P = softmax(S)`、`O = PV` 三次都把 `Sq×Skv` 写出去再读回来；
* `tiled`：`Sq×Skv` 一次都不出片上，代价是**每个 Q 块都要重读一遍 K 和 V**。

同时打印 prefill（`Sq = Skv = N`）与 decode（`Sq = 1`）两种形状的对比——它们给出的
结论完全不同，这正是第 5.4 节要讲的事。

## 运行前提

任意有 Python 3.9+ 的机器。**只用标准库**，不需要 numpy、不需要 GPU/NPU、不 import
tilelang。

## 运行命令

```bash
python3 traffic_model.py
```

## 期望输出（写作机实际输出，退出码 0）

```text
[prefill 8K]
shape: Sq=8192, Skv=8192, d=128, dtype=16 bit (2 B/element)
tiles: block_m=128, block_n=128, Q blocks=64, KV blocks=64

item                   naive           tiled
--------------------------------------------
Q read             1,048,576       1,048,576
K read             1,048,576      67,108,864
S write           67,108,864               0
S read            67,108,864               0
P write           67,108,864               0
P read            67,108,864               0
V read             1,048,576      67,108,864
O write            1,048,576       1,048,576
--------------------------------------------
total            272,629,760     136,314,880
bytes            545,259,520     272,629,760
ratio naive/tiled = 2.00x
```

```text
A. prefill-like: Sq = Skv = N        ... 每一行都是 2.0x
B. decode-like: Sq = 1, Skv = N      ... 每一行都是 ~1x
```

decode 那一半是这一节的**结论**，不是反例：当只有一行 query 时，`Sq×Skv` 本身就只是
`1×Skv`，朴素版根本不浪费流量。**decode 的问题是并行度不够，不是流量太大**——
这就是 split-KV 存在的原因。

## 判据

* 程序正常退出（退出码 0）；
* prefill 一行里 `S write` / `S read` / `P write` / `P read` 四列在 `tiled` 侧全为 0；
* `naive` 的总数等于 `4·Sq·Skv + 2·(Sq+Skv)·d`。

## 注意事项（这是模型，不是测量）

* 这里数的是**元素个数**，不是字节、更不是时间。要换算成字节乘以 dtype 宽度
  （脚本里按 fp16 = 2 B 打印）。
* 模型**刻意简单**：它假设每次访问都真的走 HBM、没有缓存、没有 TMA 的一次搬运多块、
  也没有考虑 `N×N` 根本放不进 SRAM 这件事。所以它给出的倍数（prefill 2.0x）比真实
  硬件上的收益小得多，**不能当作性能预测**。
* 它唯一的作用是：让「三趟变一趟」这句话有一个可以对数的形式。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0，
输出即上表。**没有在任何加速器上测量过任何东西。**
