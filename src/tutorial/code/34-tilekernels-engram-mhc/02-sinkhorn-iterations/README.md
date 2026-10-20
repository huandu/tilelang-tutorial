# 示例 02：两处 Sinkhorn——同一个词，两件不同的事

## 目的

TileKernels 里有**两个**叫 Sinkhorn 的算子族，很容易混为一谈：

* `mhc/sinkhorn_*`：对每个 token 的 `4 × 4` mix 矩阵先做 **row softmax**，
  再交替做**行和 / 列和归一化**，把它推向**双随机矩阵**（doubly stochastic）。
* `engram/sinkhorn_*`：对一张 `(num_rows, head_dim)` 的大矩阵交替做 **L2 行缩放 /
  列缩放**，并被拆成 `momentum → step → step_reduce → finalize` 四个 kernel。

脚本把两个过程都在 CPU 上跑一遍，好让读者看出「迭代结构」「归一化用什么范数」
「有没有 softmax」「状态大小」这四点上的差别。

## 运行前提

* Python 3.9+、PyTorch（CPU 即可）。

## 运行命令

```bash
python3 sinkhorn_iterations.py
```

## 期望输出

在写作机上的**实际输出**（节选；完整输出见运行结果）：

```text
== mHC Sinkhorn: 4 x 4, softmax then row/col normalisation ==
  repeat   max|row_sum-1|  max|col_sum-1|  max|X_k - X_{k-1}|
       1        4.563e-01       1.729e-06                 nan
       3        1.539e-01       1.192e-06           1.518e-01
      10        5.696e-03       1.073e-06           6.599e-02
      30        1.431e-06       1.073e-06           2.646e-03

  the row softmax is what keeps large logits finite:
  offset   +100: finite=True  max|row-1|=1.214e-04
  offset   -100: finite=True  max|row-1|=1.214e-04

== Engram Sinkhorn: alternating L2 row / column scaling ==
  matrix (3, 4) fp32; row_scale (3,); col_scale (4,)
  iterate = row_scale[:, None] * matrix * col_scale
  step   max|row L2 - 1| after row step        col L2 before col step
     0                      5.960e-08   [1.02401, 0.48524, 1.10937, 0.6966]
     2                      1.192e-07   [0.85121, 0.85212, 0.91841, 0.84014]
     7                      1.192e-07   [0.86589, 0.86586, 0.86646, 0.8659]
  after a few rounds the row / column scale profiles stop changing:
    row L2 before finalize: [1.15488, 1.15442, 1.1548]
    row L2 after  finalize (alpha = sqrt(head_dim) = 2.0): [2.0, 2.0, 2.0]
```

## 怎么读这些数字

* **mHC 版**：`row softmax` 那一步让行和恒为 1（列和还有偏差），随后每多迭代一轮，
  行列和就同时更接近 1。`repeat = 10` 时最大偏差约 `5.7e-3`；`repeat = 30` 时到
  `1.4e-6`。上表里的 `max|col_sum-1| ≈ 1.1e-6` 并不是误差没收敛，而是 `eps = 1e-6`
  本身留下的固定偏移——kernel 里每一步都是除以 `(sum + eps)`（
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:39,43,48,52`）。**这就是为什么 kernel 里的 `repeat`
  是个可调参数**：它决定「快」与「准」的取舍。
  另外要注意口径：**论文用的迭代次数是 `t_max = 20`**
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:405-406`），
  而本脚本与仓库测试用的是 `repeat = 10`；两者的「一次迭代」定义并不相同
  （论文一次 = 行 + 列，代码从 softmax 后的列归一化开始计数）。
  要做论文口径的对照，先读正文 §5.2 那张「论文 vs 源码」表。
* **offset ±100 两行**：即使把整个 4×4 矩阵平移到 ±100，结果仍然有限。原因是
  **先做行 softmax**（内部减掉行最大值），这与你熟悉的 online softmax 稳定性
  是同一个道理。
* **Engram 版**：`iterate = row_scale ⊗ matrix ⊗ col_scale` 是「矩阵外积缩放」，
  归一化用的是 **L2 范数**而不是和。每做一次行步，行 L2 变成 1；再做一次列步，
  列 L2 变成 1，行 L2 又偏离 1。交替几轮后行/列的缩放分布稳定下来（互为倒数），
  最后由 `finalize` 用一个 `alpha` 把全局因子定死——所以你会看到「finalize 之后
  每一行的 L2 范数正好等于 `alpha = sqrt(head_dim)`」。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11、torch 2.1.1 上实际运行过。
脚本里的两个 `*_ref` 函数是从 corpus 的
`tile_kernels/torch/mhc.py:8` 与 `tile_kernels/torch/engram.py:39,66` **原样抄来**
的，所以这里的数字就是仓库自带 CPU 基线的行为。**没有跑过 kernel。**

## 已知局限

* 小矩阵（`3 × 4`、`2 × 4 × 4`）不代表真实规模。mHC 的真实矩阵是每个 token 一个
  `4 × 4`（`hc = 4`）；Engram 的矩阵在测试里是 `10⁵ × 5120` 量级
  （`tests/engram/test_engram_sinkhorn.py:98`）。
* L2 交替缩放的收敛速度与矩阵本身有关；脚本只展示「它会收敛」和「`finalize` 做什么」，
  不给收敛率。
* Engram 版在真实训练里挂在哪个参数上（哪一步用哪个 kernel 组合），TileKernels
  仓库里没有调用方，本章只能给出**源码 → 行为推断**，详见正文 §5.3 与
  `../references.md` 的「待补资料」。
