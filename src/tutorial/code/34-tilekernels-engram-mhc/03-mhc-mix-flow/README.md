# 示例 03：mHC 的 mix 拆分与应用

## 目的

mHC 前向/反向在训练里被拆成六个以上的 kernel。这个脚本用 CPU 把它们串成一条
可读的数据流，回答：

* 每个 token 的 `mixes` 张量为什么是 `24` 个分量？`24 = 2 × 4 + 4 × 4` 各是什么？
* `pre_apply_mix`、`post` 分别在算哪个公式？
* `expand` 与 `head_compute_mix` 这两条“边界”算子在干什么？

公式与形状全部抄自 corpus 的纯 PyTorch 基线与 kernel 断言：

* `mhc_pre_split_mixes_ref`、`sinkhorn_normalize_ref`、`mhc_pre_apply_mix_ref`、
  `mhc_post_ref`：`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py`
* 形状与 `mhc_mult3 = 2*mhc + mhc²`：`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:24-33`、
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/post_kernel.py:23-25`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/expand_kernel.py`

## 运行前提

* Python 3.9+、PyTorch（CPU 即可）。

## 运行命令

```bash
python3 mhc_mix_flow.py
```

## 期望输出

在写作机上的**实际输出**：

```text
== shapes ==
  residual (M branches)    (3, 4, 64) torch.bfloat16
  mixes (per token)        (3, 24) torch.float32
  layer output             (3, 64) torch.bfloat16
  MHC3 = 2*MHC + MHC^2 = 2*4 + 16 = 24

== split + sinkhorn ==
  pre  (3, 4, 1)  post (3, 4, 1)  comb (3, 4, 4) -> normalized
  row sums of comb: [0.999999, 0.999999, 0.999999, 0.999999]
  col sums of comb: [0.999999, 0.999999, 0.999999, 0.999999]

  layer_input = sum_m pre_m * residual_m : (3, 64) torch.bfloat16
  new residual = post * layer_output + comb^T @ residual : (3, 4, 64)

== residual-stream recurrence, layer by layer ==
  layer 0: |layer_input| mean = 0.7979, |residual| mean = 0.5357
  layer 1: |layer_input| mean = 1.0416, |residual| mean = 0.7936
  layer 2: |layer_input| mean = 1.5658, |residual| mean = 0.7936

== the two boundary ops ==
  expand   fwd: x[n,h] -> [4] copies;  bwd: sum over the 4 branches
  expanded (3, 4, 64) -> collapsed (3, 64) (same as the input)
  head_compute_mix: sigmoid(x * scale + base) + eps  (eps keeps the gate > 0)
```

## 怎么读这些结果

* `pre` / `post` 各是 **每个 token 4 个数**（一个 head 一个缩放），`comb` 是
  **4 × 4 的连接权重**——这就是 Engram 论文 §2.4 说的“M 条并行分支、信息流由
  可学习连接权重调制”落到代码里的样子。
* `comb` 经过 Sinkhorn 之后，行列和都是 `1`（误差 `1e-6` 级，来自 `eps`）。
  这就是 mHC 的“manifold constraint”：mix 矩阵被约束在**双随机矩阵流形**上。
* 每层反向/重算时，残差流的递推是
  `res_{l+1}[o] = post[o] · out_l + Σ_m comb[m, o] · res_l[m]`。
  `mhc_post` 与 `mhc_multilayer_recompute` 实现的是同一个递推，后者只是把它
  在**多层上循环**并重算 `layer_input`（见正文 §6.5）。
* `expand` 的前向是把单流特征复制成 4 份，反向是把 4 份梯度加起来；
  `head_compute_mix` 是 `sigmoid(x·scale + base) + eps`，`eps` 保证 mix 严格大于 0。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11、torch 2.1.1 上实际运行过。
`pre/post/comb` 的拆分与 `post` 的收缩顺序与 corpus 基线逐项对齐；脚本跑出来的
“行列和 = 1”用的是同一份 `sinkhorn_normalize_ref`。**没有跑过 kernel，也没有对比过
`mhc_pre_big_fuse` 的融合实现。**

## 已知局限

* 脚本跳过了 `prenorm GEMM` 与 `reduce_partials_and_rmsnorm`（它们需要真实 GEMM
  输出），也就是把 `mixes` 直接喂给 `split_mixes`。真实链路是
  `residual → GEMM → split-K 部分和 → reduce+rmsnorm → mixes`。
* 只用了 `hidden = 64`、`n = 3`；真实 `hidden` 在 1280–8192（见
  `tests/mhc/test_pre_big_fuse.py:183`）。
* 公式来源有两条：**论文**（`paper-deepseek-mhc.md` §3 式 (1)–(5)、§4.1 式 (6)、
  §4.2 式 (7)(8)）与**仓库的纯 PyTorch 基线**。两者不一致之处（`post` 的 `2σ`
  系数、`pre` 的 `eps`）已在正文 §5.2/§6.2 写明；本脚本用的是**代码**的口径。
