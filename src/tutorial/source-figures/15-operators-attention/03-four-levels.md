# 图 3：attention 的四级递进实现

对应正文第 3.1–3.5 节。要点：**每一级只改一件事，而且能说清“上一版为什么不够”。**

## ASCII 版本

```text
① 朴素版：把 N×N 物化
   ┌──────────────────────────────────────────────────────────────┐
   │ 一个线程块负责一个 query 行块：                               │
   │   把整个 K、V 搬进 shared；算出整块 S；整行 softmax；再乘 V    │
   │                                                              │
   │ 不够用因为：                                                  │
   │   · S_shared 是 block_M × seq_kv，seq_kv=8K 时就已经 2 MB     │
   │   · 必须算完整行才能开始输出，算和写被硬性串起来               │
   └──────────────────────────────────────────────────────────────┘
                                 │ 改一件事：把 N×N 拆成块 + online softmax
                                 ▼
② 分块 + online softmax
   ┌──────────────────────────────────────────────────────────────┐
   │ for k in 0 … ceil(skv/block_n):                              │
   │     S_blk = Q_blk · K_blkᵀ          （fragment，不落地）      │
   │     m_new = max(m_prev, rowmax(S_blk))                       │
   │     c = exp2(m_prev·s − m_new·s)    （≤ 1 的重标度因子）      │
   │     ℓ = c·ℓ + rowsum(P);  O = c·O + P·V_blk                  │
   │ O_final = O / ℓ                                              │
   │                                                              │
   │ 消掉了：4N² 的 HBM 流量（S/P 一次都不写出）                    │
   │ 不够用因为：                                                  │
   │   · copy 与 gemm 前后排着，没有重叠（num_stages=1）            │
   │   · Q 只在块内复用；K/V 每个 Q 块重读一遍（模型里 +2·(N/bm)·N·d）│
   └──────────────────────────────────────────────────────────────┘
                                 │ 改一件事：加流水 + 寄存器复用
                                 ▼
③ 加流水与 shared memory 复用
   ┌──────────────────────────────────────────────────────────────┐
   │ + for k in T.Pipelined(loop_range, num_stages=N)             │
   │ + T.gemm(..., policy=T.GemmWarpPolicy.FullRow)               │
   │ + T.copy(acc_s, acc_s_cast)（softmax 结果复用 fp32 缓冲）      │
   │                                                              │
   │ 换来的：第 k+1 块的 K/V 搬运与第 k 块的计算重叠                 │
   │ 代价：K_shared / V_shared 各占 num_stages 份                   │
   │       block_M=block_N=dim=128、fp16 时每份 32 KB               │
   │       ns=2 → 160 KB；ns=3 → 224 KB（还含 Q_shared 32 KB）      │
   │ 不够用因为：                                                  │
   │   · 形状写死：没有 causal、没有 past_len、没有 varlen、没有 GQA │
   └──────────────────────────────────────────────────────────────┘
                                 │ 改一件事：对齐官方实现
                                 ▼
④ 与官方 example_mha_fwd_bhsd.py 对齐
   ┌──────────────────────────────────────────────────────────────┐
   │ (1) scale 折进 log2(e)，exp → exp2                            │
   │ (2) past_len = seq_kv − seq_q，q_idx = bx*block_M + i + past_len│
   │ (3) causal 的循环上界：ceil(((bx+1)*block_M + past_len)/block_N)│
   │ (4) 越界 key 也要 mask（T.copy 补的 0 是「真实得分 0」）         │
   │ (5) epilogue 经 O_shared 再写回全局，避免非合并访问             │
   └──────────────────────────────────────────────────────────────┘
```

## 对照表

| | ① 朴素 | ② 分块 online | ③ + 流水/复用 | ④ 官方 |
| --- | --- | --- | --- | --- |
| `S`、`P` 落地 HBM | 是 | 否 | 否 | 否 |
| 遍历 key 的趟数 | 3 | 1 | 1 | 1 |
| 搬运/计算重叠 | 否 | 否 | 是 | 是 |
| warp 划分 | 未指定 | `Square` | `FullRow` | `FullRow` |
| causal | 无 | 无 | 无 | 循环裁剪 + mask |
| `seq_kv > seq_q` | 无 | 无 | 无 | `past_len` |
| epilogue | 直接写 | 直接写 | 直接写 | 经 `O_shared` |

## 读图要点

1. **每一级都只改一件事。** ② 改的是“中间结果不出片上”，③ 改的是“搬运与计算的
   时间关系”，④ 改的是“覆盖真实形状”。混在一起改，出错时不知道是哪一处引起的。
2. **② 和 ③ 解决的是不同的问题。** ② 省的是 HBM 流量，③ 省的是时间轴上的等待。
   一个 kernel 可以只做到 ②（正确但慢），也可以只做到 ③ 的表层（`T.Pipelined` 加了
   但 `S`/`P` 还在落地。那就没有意义）。
3. **④ 的每一处都能在 `code/04-block-flash-attention` 里被数值验证**（full / causal /
   `past_len` / window+sink / varlen 五个用例）。

## 来源

* 代码对照对象：`_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py`
  （五处差别的行号见正文 3.4 节）。
* 片上占用的算法：`block_M × dim × 2 B` 与 `block_N × dim × 2 B` 各乘以份数；
  份数由 `num_stages` 决定（第 11 章 4.4 节）。
* 数值验证：本章 `code/04-block-flash-attention/block_flash_attention.py` 的输出。
* 本图不含任何硬件实测数据。
