# 示例 01：Engram 门控的 shape 走查与融合恒等式

## 目的

回答两个问题：

1. `engram_gate_fwd` 的输入输出到底长什么样、哪一维是 head、哪一维是被
   `hidden_size` 归一化掉的归约维？
2. 「把 RMSNorm 的权重先乘成一个 `weight_fused`，再在 kernel 里只做一次累加」——
   这个融合在**代数上**是否与原式等价？

第 2 点是本章 §3 讲的融合设计的核心。脚本把两条路径都算一遍：

* `gate_naive`：先把 `RMSNorm(x, w_h)` 与 `RMSNorm(k, w_e)` 分别写成张量，再点积；
* `gate_fused`：用 `weight_fused = w_h * w_e`，一次累加得到 `Σ x·w_h·w_e·k`。

两者的数学来源是 `RMSNorm(x, w_h)ᵀ · RMSNorm(k, w_e) = rstd_x · rstd_k · Σ x·w_h·w_e·k`。

## 运行前提

* Python 3.9+、PyTorch（CPU 即可，不需要 GPU/NPU）。

## 运行命令

```bash
python3 engram_gate_shapes.py
```

## 期望输出

在写作机（macOS Apple Silicon、Python 3.11、torch 2.1.1、CPU）上的**实际输出**：

```text
== shapes ==
  hidden_states                (5, 4, 256)  torch.bfloat16
  kv                           (5, 5, 256)  torch.bfloat16
  key k  = kv[:, :hc_mult]     (5, 4, 256)  torch.bfloat16
  value v = kv[:, hc_mult]     (5, 256)  torch.bfloat16
  weight_hidden                (4, 256)  torch.bfloat16
  weight_embed                 (4, 256)  torch.bfloat16
  weight_fused (fp32)          (4, 256)  torch.float32

== fusion identity: unfused vs fused ==
  max|gate_naive - gate_fused| = 1.192e-07
  max|dot_naive  - dot_fused | = 2.384e-07

  output               shape (5, 4, 256)  torch.bfloat16
  gate_score (0, :)    [0.2967078387737274, 0.5653913021087646, 0.679608166217804, 0.7557258009910583]
  raw_dot   (0, :)     [-11.916980743408203, 1.1073100566864014, 9.047382354736328, 20.40825080871582]

== tensors saved for backward (shape, bytes) ==
  dot         (5, 4) torch.float32      80 B
  gate_score  (5, 4) torch.float32      80 B
  rstd_x      (5, 4) torch.float32      80 B
  rstd_k      (5, 4) torch.float32      80 B
  total       320 B for 5 tokens x 4 heads

== naive path: what the fusion removes ==
  normed_x + normed_k intermediates: 40960 B (fp32), written then read
  fused path keeps only 3 per-thread accumulators (x^2, k^2, x*w*k)

== how many bytes does the fused kernel touch (bf16 in, bf16 out)? ==
  read  x + kv :   23040 B
  write output :   10240 B
  weight_fused : 4096 B (fp32, resident)
```

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11、torch 2.1.1 上实际运行过，
输出即上面这段。`max|gate_naive - gate_fused| < 1.3e-7`（fp32 精度下）证明融合恒等式
成立。**没有在 GPU/NPU 上运行过任何 kernel。**

## 已知局限

* 脚本用的是 `hidden_size = 256`、`num_tokens = 5`；真实 kernel 只针对
  `hidden_size ∈ {4096, 7168}` 调过参数（`engram_gate_fwd_cuda.py:31`），且要求
  `hidden_size` 能被 128（昇腾）或对应的 `blk_d`（CUDA）整除。
* 「融合前后各多少字节」是**按元素个数算的记账**，不是测量。真实收益还取决于
  cache、寄存器压力与是否真的多跑了一次 kernel。
* 脚本没有实现 `clamp_value` 边界的梯度、也没有实现反向；反向见正文 §4。
