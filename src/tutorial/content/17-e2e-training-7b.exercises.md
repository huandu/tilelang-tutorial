# 第 17 章 习题

难度标记 `★ / ★★ / ★★★`；标「需硬件」的题目需要 GPU/NPU 才能完成到「运行验证」
一级，其余可在 CPU 上完成。所有题目只用本章及之前章节的知识即可作答。

---

### E1 ★（概念 · 不需要硬件）

Qwen2.5-7B 的参数量是 7.61B。请**不用运行脚本**，手算出它在「bf16 权重 + bf16 梯度 +
fp32 Adam（m、v）+ fp32 主权重」配置下的总显存（单位 GiB），并说明为什么单张 80GB 的卡
放不下。

### E2 ★（概念 · 不需要硬件）

`C ≈ 6·N·D` 里的「6」是怎么来的？在什么情况下这个估算会明显偏低？举一个具体的例子。

### E3 ★（概念 · 不需要硬件）

对比「小规模预训练复现」与「从开源权重继续训练」两条路线：分别写出它们各适合什么目标、
主要成本在哪、最大的风险是什么。为什么本教材建议先用前者验证链路、再用后者做出效果？

### E4 ★（代码阅读 · 不需要硬件）

读 `code/01-plan-ledger/train_plan.py` 的 `memory_bytes`。问题：

1. 权重、梯度、Adam m/v、fp32 主权重各自占多少字节/参数？
2. 如果把 `fp32_master=False`，总量变成多少 GiB？
3. 为什么「优化器状态」是显存优化最优先的目标？

### E5 ★★（代码阅读 · 不需要硬件）

读 `code/02-data-pipeline/pack_tokens.py`。问题：

1. `pack` 为什么丢弃最后一个不满 `block_size` 的尾块，而不是 padding 补齐？
2. `shuffled` 为什么用固定容量的 buffer 而不是「收集全部再打乱」？
3. 如果 `--seed` 相同但 `--shuffle-buffer` 不同，输出会一样吗？为什么？

### E6 ★★（代码阅读 · 不需要硬件）

读 `code/03-tilelang-norm-torch/rmsnorm_tl.py`。问题：

1. `@tilelang.jit(out_idx=[2])` 里的 `2` 指的是什么？换成 `[-1]` 等价吗？
2. `RMSNormTileLang.forward` 里 `ctx.save_for_backward(x, w)` 保存的是什么？为什么不在
   反向里重算？
3. 如果把 `backward` 改成 `return None, None, None`，会发生什么？

### E7 ★★（概念 · 不需要硬件）

为什么 TileLang（v0.1.15）不能自动为 `torch.autograd` 提供反向？请给出至少两种「补上
反向」的方案，并各说出一个优缺点。

### E8 ★★（动手 · 不需要硬件）

当前 `pack_tokens.py` 的 packing 会让同一条序列跨越文档边界。请改造它，增加一个
`--no-cross-doc` 选项：当一篇文档放不下时，把它整体挪到下一条序列，前面用
「丢弃 + 记录浪费 token 数」的方式处理（不要 padding）。说明这样做的代价。

### E9 ★★（动手 · 不需要硬件）

用 `code/01-plan-ledger/train_plan.py` 估算 Qwen2.5-**0.5B**（按名义 0.5B 参数）在
`D = 5e9` token 上的显存与算力账本。给出你用的命令、关键输出，并回答：0.5B 能否在单张
40GB 卡上做全量 Adam 训练（不考虑激活值）？

### E10 ★★★（动手 · 不需要硬件）

为 `code/06-train-loop/train_step.py` 增加一个自检：在极小模型上，比较「`micro_batch=B,
grad_accum=1`」与「`micro_batch=B/G, grad_accum=G`」两种配置下**同一步**的梯度（例如
`lm_head.weight.grad` 的范数）。要求给出通过判据并解释为什么允许少量误差。

### E11 ★★（代码阅读 · 不需要硬件）

对比 `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:39-66`（NV）与
`_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:44-70`（昇腾）。问题：

1. 两者做归约的方式有什么不同？
2. 昇腾版本里 `T.annotate_buffer_versions` 和 `T.SimtVF` 是用来干什么的？
3. 这些差异说明「同一个算子跨硬件复用」时最需要关注什么？

### E12 ★★★（设计题 · 不需要硬件）

你要把一个 attention 算子换成 TileLang 实现，并保证训练可用。请写出一份验收清单，
至少覆盖：前向数值对齐、反向从哪里来、反向数值对齐、端到端 loss 对齐、以及吞吐不下降。
每一条都要给出**具体的检查方法**与**通过门限**。

### E13 ★★（概念 · 不需要硬件）

训练日志显示 GPU 利用率周期性掉到接近 0，但 kernel 本身已经优化过。按本章 §4.4，
你应该优先排查哪些环节？为什么不应该先去改 kernel？

### E14 ★★★（动手 · 需硬件）

在一块 NVIDIA GPU 上完成 M1–M4 中的至少 M3 与 M4：先用 `--from-hf` 训练一个小模型
（如 Qwen2.5-0.5B）若干步，跑出 loss 曲线；再把 RMSNorm 换成
`code/03-tilelang-norm-torch/rmsnorm_tl.py` 的实现，比较训练曲线与吞吐。记录你的
硬件、命令、期望输出与实测结果。
