# 示例 4：本章的参考实现（五个变体逐个对照）

## 目的

这是本章的**参考实现**：把官方 kernel 的主循环（`example_mha_fwd_bhsd.py:40-100`）
逐行搬到 CPU 上，用同一个循环结构覆盖五个变体，每个变体都与一个**独立的、
直接建 mask 的精确 softmax** 对照。正文里的每一处「这一版改了什么」都应该能在这里
找到对应的数值结论。

| 用例 | 对应官方案例 | 检查什么 |
| --- | --- | --- |
| 1 | `example_mha_fwd_bhsd.py` | 五种 `block_m × block_n` 组合都能复现精确结果 |
| 2 | 同上（`past_len`） | causal + KV cache 前缀；`past_len = 0/5/12` |
| 3 | `example_mha_sink_fwd_bhsd.py` | sliding window + sink；**同时验证 `check_inf` 是不是必需** |
| 4 | `example_mha_fwd_varlen.py` | `cu_seqlens` 打包 + 短样本的写回 guard |

## 运行前提

Python 3.9+ 与 numpy。不需要 GPU/NPU、不 import tilelang。

## 运行命令

```bash
python3 block_flash_attention.py
```

## 期望输出（写作机实际输出，退出码 0）

```text
case 1: full attention (no mask)
   block_m x block_n       max abs err
               4 x 4         4.441e-16
               8 x 5         3.331e-16
             20 x 20         4.441e-16
               8 x 8         3.331e-16
              4 x 20         4.441e-16

case 2: causal, with and without a KV-cache prefix (past_len)
past_len=  0 (Sq=24, Skv=24): max abs err = 4.441e-16
past_len=  5 (Sq=24, Skv=29): max abs err = 4.996e-16
past_len= 12 (Sq=24, Skv=36): max abs err = 4.441e-16

case 3: sliding window + attention sink, with and without check_inf
window=8, sink=3.5
  with    check_inf : max abs err = 1.665e-16
  without check_inf : max abs err = nan   (NaN in 48 entries)
  causal + sink, no window, check_inf off: max abs err = 2.220e-16

case 4: varlen packing (cu_seqlens) vs. one kernel per sample
lengths=[7, 19, 4, 13], packed total=43, max_seqlen=19
packed kernel vs per-sample reference: max abs err = 4.441e-16
unwritten padding rows (should be 0): 0
```

## 判据（脚本内部有断言，失败即非 0 退出）

* 用例 1/2/4：误差 `< 1e-12`；
* 用例 3：开 `check_inf` 时误差 `< 1e-12`，关 `check_inf` 时**必须出现 NaN**；
* 用例 3 的第二行是关键对照：**causal + sink 而没有 window 时，`check_inf` 关掉也
  不会出错**——因为 causal 的对角线永远可见，运行最大值从第一块起就是有限的。
  「哪些形状才需要 `check_inf`」由此变成一个可验证的问题，而不是经验之谈。

## 注意事项

* 这里的「精确参考」用的是**直接构造 mask + 完整 softmax**，没有用到 online softmax，
  所以两者的独立性能互相印证第 2.4 节的证明。
* 本示例为了可读性把 `block_m` 之外还加了 `q_offset` 参数（用例 4 需要在同一批数据上
  分块调用）。官方实现是用 grid 的 `bx` 表达同一件事。
* **它不是 kernel**：没有 layout、没有多缓冲、没有 warp 划分，因此**不能**从它推出
  任何性能结论。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 + numpy 1.24.1 上实际运行过，
退出码 0，所有断言通过。**没有运行过任何 TileLang kernel。**
