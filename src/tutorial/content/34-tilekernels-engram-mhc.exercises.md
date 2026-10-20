# 第 34 章 习题

说明：

* 难度：`★`（读完正文即可作答）/ `★★`（需要综合本章多节或动手）/ `★★★`（需要
  逐行核对源码、多步推理）。
* 「需要硬件」一栏里，`否` 表示只需要一台能跑 Python 的机器；本章**没有**必须
  用加速器才能做的题——所有需要真机的验证都标为「可选」。
* 所有题目只依赖第 1–34 章的知识；涉及第 35 章（DeepGEMM-Ascend）的地方会显式提示。
* 题目里提到的 `code/0X` 都指本章的示例目录；正文与答案里的源码引用一律写成完整的
  corpus 路径（`../tilelang-corpus/14-deepseek-ascend-stack/raw/...`），
  在站点上会渲染成可点开的上游链接。

---

## E1 概念题 ★ 不需要硬件

请用几句话回答：

1. 论文为什么把 Engram 称为「与 MoE 互补的第二条稀疏轴」？两条轴各自「稀疏」在哪？
2. 「确定性寻址（deterministic addressing）」在这里具体指什么？为什么它能支撑
   「把表卸载到 host memory」？
3. 论文说的 `O(1)` 查表，指的是「取几行的数量不随表增大」还是「访存延迟是常数」？
   请各举一个本章里的证据。

---

## E2 概念题 ★★ 不需要硬件

本章反复强调「论文结论」与「仓库能证明的事」要分开。

1. 论文给出的 U 型分配律（最优在 `ρ ≈ 75%–80%`）能不能用
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py` 验证？为什么？
2. `Engram/README.md` 里哪一句话支持你的结论？
3. 「乘法-XOR 哈希」这件事，论文与 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_cuda.py`
   是一致的吗？请指出两边的说法/实现。

---

## E3 概念题 ★ 不需要硬件

写出 Engram 门控的完整数学式（用 `x`、`k`、`v`、`w_h`、`w_e`、`d`、`eps`、
`clamp_value` 表示），并回答：

1. 为什么 `w_h` 与 `w_e` 可以先乘成一个 `weight_fused`？请用一行推导说明。
2. `signed-sqrt` 与 `clamp_value` 各自解决什么数值问题？
3. `alpha` 的取值范围是什么？为什么 `alpha → 0` 表示「抑制这段记忆」？

---

## E4 代码阅读题 ★★ 不需要硬件

下面这段来自 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py`：

```python
for i_b in T.Serial(1, num_blk):
    phase = i_b % 2
    prev_phase = (i_b - 1) % 2
    T.async_copy(hidden_states[i_s, pid_h, i_b * blk_d : (i_b + 1) * blk_d], x_smem[i_b * blk_d : (i_b + 1) * blk_d])
    T.async_copy(kv[i_s, pid_h, i_b * blk_d : (i_b + 1) * blk_d], kv_smem[phase, :])
    T.ptx_wait_group(2)
    for i_sub in T.Serial(sub_blks):
        sub_base = (i_b - 1) * blk_d + i_sub * reduce_blk
        for i_k in T.vectorized(vec_size):
            x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
            k_local[i_k] = kv_smem[prev_phase, i_sub * reduce_blk + thread_idx * vec_size + i_k]
        for i_k in T.vectorized(vec_size):
            w_local[i_k] = weight_fused[pid_h, sub_base + thread_idx * vec_size + i_k]
        for i_k in T.serial(vec_size):
            rstd_x_local[0] += x_local[i_k] * x_local[i_k]
            rstd_k_local[0] += k_local[i_k] * k_local[i_k]
            gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]
```

1. 这段在算哪三个标量？分别对应数学式里的哪一项？
2. 它读的是 `weight_fused` 而不是 `weight_hidden` / `weight_embed`，为什么？如果改成
   分别读两个权重再相乘，会多出什么开销？
3. `T.ptx_wait_group(2)` 在这里等的是哪几次拷贝？为什么是 `2` 而不是 `0`？
4. `i_b` 从 1 开始而不是 0，为什么？（提示：看 prologue 与 `prev_phase`。）

---

## E5 代码阅读题 ★★★ 不需要硬件

把 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:90-195` 与
`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:76-222` 对照着读，完成下面表格：

| 关注点 | CUDA 版怎么写 | 昇腾版怎么写 |
| --- | --- | --- |
| 一个任务处理几个 head | ? | ? |
| 多缓冲怎么实现 | ? | ? |
| 三个累加量的实现 | ? | ? |
| 把标量闸门广播到每条向量 | ? | ? |
| 输出怎么回到显存 | ? | ? |

然后回答：这两份实现里，**哪一部分是「完全一一对应」的，哪一部分是「各自机器
独有的」**？为什么说这就是「一份算法、两个后端」的含义？

---

## E6 计算题 ★★ 不需要硬件

设 `T = 8192`、`hc = 4`、`d = 4096`，数据为 bf16（2 字节）。

1. `hidden_states`、`kv`、`output` 各是多少字节？
2. 门控前向的「必读 + 必写」是多少字节？（只算前向，不算 `save_for_backward`）
3. `save_for_backward` 写出的四个 fp32 张量是多少字节？与上面的总量相比是几分之几？
4. 如果**不融合**，把 `normed_x`、`normed_k` 当作 fp32 中间张量各写一遍再读一遍，
   会多出多少字节？用 `code/01` 的口径说明这个估算忽略了什么。

---

## E7 概念题 ★★ 不需要硬件

关于反向的三个 kernel：

1. `engram_gate_bwd` 输出的 `grad_w_partial` 的第一维为什么是「持久块数」而不是
   `hc_mult`？
2. `grad_w_reduce` 做的两件事是什么？请写出与
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/engram/test_engram_grad_w_reduce.py:19-22` 等价的三行公式。
3. 为什么不干脆在 `engram_gate_bwd` 里用原子加直接累加参数梯度？请给出本章提到的
   至少一条理由。
4. `main_grad` 的存在如何影响 `grad_w_reduce` 的语义（写入 vs 累加）？

---

## E8 概念题 ★★ 不需要硬件

两处 Sinkhorn：

1. 各自用什么范数归一化？各自的状态是什么、多大？
2. `mhc_sinkhorn_fwd` 为什么要先做行 softmax？
3. `engram_sinkhorn_step` 为什么要拆出一个 `step_reduce`？
4. 为什么两处都要把迭代放在 kernel 内部，而不是每轮启动一个 kernel？

---

## E9 代码阅读题 ★★★ 不需要硬件

读 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_cuda.py:44-106`，回答：

1. 这个 kernel 的 CTA 数是多少？每个 CTA 负责多少数据？
2. 它用 `T.get_thread_binding() < 32` 把线程分成了哪两组？两组各做什么？
3. 它的注释把代码分成了四段（`_mhc_reduce_partials_and_rmsnorm_fwd` 等）。
   请把这四段与正文 §6.2 的四个算子名字对应起来。
4. 昇腾版的注释说它「Mirrors the arithmetic of the individual Ascend reference
   kernels … op-for-op so the fused result is bitwise identical」。这句话对
   「融合 kernel 的正确性怎么保证」给出了什么启发？
5. 论文 §4.3.1（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:413-502`）
   把融合分成四条：式 (14)–(15) 一个 kernel、式 (16)–(18) 一个 kernel、
   式 (19) 一个 kernel、外加两个「应用映射」的 kernel。请指出
   `pre_big_fuse` 覆盖了其中哪几条、论文里被拆开而这里被合起来的是哪些，
   并说明这样「多融一层」在什么条件下是安全的。

---

## E10 概念题 ★★ 不需要硬件

关于 `multilayer_recompute`：

1. 它省下了什么？为什么值得多做一遍前向？
2. 它为什么要把「一串张量」做成指针表传给 kernel？`T.ptr` + `T.make_tensor`
   的代价是什么？
3. 代码里为什么要把新残差先 `cast` 回 bf16 再转回 fp32？
4. `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_multilayer_recompute.py:99-108` 用 IO 字节数说明收益。
   请说明这个记账高估/忽略了哪些真实开销。
5. 论文 §4.3.2 的 Table 3 把「每层都存」「每 L_r 层存一次」「块内重算」
   分成了三类。请在 kernel 的七个参数里各找一个对应物，并指出论文式 (20) 给出的
   最优块长 `L*_r ≈ sqrt(nL/(n+2))` 在本章的实现里由谁决定。

---

## E11 设计题 ★★ 不需要硬件

下面三个算子，分别应该用 TileLang 写，还是像 DeepGEMM 那样下钻？请对每一个给出
「判据 → 结论」，并说明如果反过来选会付出什么代价。

1. 对每个 token 的 `4 × 4` mix 矩阵做 10 轮 Sinkhorn，再把它乘回 4 条向量分支；
2. `[n, 24] × [24, 28672]` 的 prenorm GEMM（`N = 24`，K 很大）；
3. 一个 `int32` TopK，需要位运算与压缩比较来压低常数（提示：第 37 章会展开，
   此处只需给判断逻辑）。

做完第 2 题之后，请核对论文 §4.3.1 里的那句话：
「We efficiently implement the majority of kernels (**excluding Eq. (14) to (15)**)
using TileLang」（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:502`）。
论文的划分与你的判断一致吗？如果不一致，是谁漏掉了什么约束？

---

## E12 动手题 ★★ 不需要硬件

用 `code/02-sinkhorn-iterations/sinkhorn_iterations.py` 做实验：

1. 把 mHC 那部分的 `eps` 从 `1e-6` 改成 `1e-3`，观察 `max|row_sum−1|` 的下限
   怎么变。解释这个下限从哪来。
2. 把 `repeat` 固定成 10，把输入矩阵的 scale 从 `2.0` 改成 `0.1`，
   收敛速度变了没有？为什么行 softmax 会说「我不管输入的整体尺度」？
3. 把 Engram 那部分的 `alpha` 从 `sqrt(head_dim)` 改成 `1.0`，
   `finalize` 之后每行的 L2 范数变成多少？这说明 `alpha` 在控制什么？
4. 论文式 (9) 的 `t_max = 20`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:405-406`）
   与仓库默认的 `repeat = 10`（`tests/mhc/test_sinkhorn.py:19`）并不等价：
   请算出「`repeat = 10` 与 `repeat = 20` 的结果差多少」，并说明为什么不能直接把
   这两个数当成同一个口径。（想清楚之后再看 §5.2 的对照表。）

（这个脚本只依赖 PyTorch，CPU 即可。）

---

## E13 动手题 ★★ 不需要硬件

在 `code/01-engram-gate-shapes/engram_gate_shapes.py` 的基础上做两件事：

1. 把 `hidden_size` 从 256 改成 5120（生产里的量级），
   确认融合恒等式仍然成立（误差仍在 1e-5 量级内）。
2. 给 `gate_naive` 加一个「显式写出 `normed_x` / `normed_k` 再点积」的实现
   （例如把两个中间张量 `clone()` 一份），统计它比 `gate_fused` 多访问了多少字节。
   与 E6 的答案对照，说明为什么真实 kernel 的收益不完全等于这个字节差。

---

## E14 分析题 ★★★ 不需要硬件

回到论文的 U 型分配律（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:364-430`）：

1. `ρ = 100%` 与 `ρ = 0%` 分别退化成什么模型？各自缺什么能力？
2. 论文说「把 20%–25% 的稀疏容量让给 Engram 最好」。如果把这句话当成工程建议，
   它在系统上要满足什么前提？（提示：想想 §2.3 讲的 `O(1)` 与可预取，
   以及如果表不能放 host memory 会怎样。）
3. 本章反复强调这个结论是「论文自报、作者未复现」。请写出一份**最小可行的验证
   计划**：要复现这张 U 型曲线，需要什么规模的训练、什么硬件、什么判别指标？
   换句话说，为什么本书不可能在写作环境里验证它？
