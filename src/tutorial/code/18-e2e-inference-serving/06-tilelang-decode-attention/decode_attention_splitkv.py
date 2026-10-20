#!/usr/bin/env python3
"""第 18 章 · decode attention（split-KV / flash decoding）的 TileLang 骨架。

验证级别：**仅静态核对**。

* 本机没有 NVIDIA GPU，也没有昇腾 NPU，本文件**从未被编译或运行过**。
* 它存在的目的是把「decode attention 需要哪些张量、两次 launch 之间要传
  什么」写清楚，让读者能对照锁定版本的官方示例逐行读。
* 结构与 API 名对照 `_vendor/tilelang-v0.1.15/examples/flash_decoding/`：
  `example_mha_inference.py`（split + combine 两个 kernel）、
  `example_gqa_decode.py`（GQA 的 `cur_kv_head` 与逐 split 的 lse）、
  `example_gqa_decode_varlen_logits.py`（varlen 的 `cu_seqlens_k`）。

与官方示例的差别（都是为了把教学要点凸出来）：

1. 只做 **decode**（`seqlen_q = 1`），query 方向没有分块；
2. KV 用 **varlen** 表示（`cu_seqlens_k`），不假设每个 batch 一样长；
3. 省掉 autotune 装饰器与 TMA 细节，只留数据流；
4. split 的边界用「按比例切」而不是「按 floor 等分」，避免短序列被切出空段。

在有卡机器上如何验证：见本目录 `README.md`。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[6])
def gqa_decode_splitkv(
    batch: int,
    heads: int,
    kv_heads: int,
    total_kv: int,
    max_kv: int,
    dim: int,
    num_splits: int = 8,
    block_N: int = 128,       # 每个 split 一次搬多少 KV
    block_H: int = 64,        # 一次处理多少个 query head
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """一次 decode step 的 GQA attention：split 段 + combine 段。

    参数
    ----
    batch / heads / kv_heads
        形状。``heads // kv_heads`` 是 GQA 的 group 数，也就是「一份 KV 被
        几个 query head 复用」——Qwen2.5-7B 是 28 // 4 = 7。
    total_kv
        varlen 打包后的 KV 总长度（整个 batch 拼在一起）。
    max_kv
        batch 里最长的 KV 长度，**编译期常量**：grid 与中间张量的形状都要
        用到它。不同 max_kv 会触发重新编译（第 7 章的 JIT 缓存）。
    num_splits
        KV 方向切几段。decode 时只有 1 个 query token，``batch × heads``
        个并行块常常喂不饱机器，靠这一维把并行度乘上去。

    约束
    ----
    ``num_splits`` 必须满足「每个 split 至少覆盖一个 KV」，
    即 ``num_splits <= min_kv``；否则会有 split 拿到空区间、epilogue 里
    出现 ``0 / 0``。真实框架在调度层保证这一点（不满足就退化成 num_splits=1）。
    """
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)：把 exp 变成 exp2
    kv_group_num = heads // kv_heads
    valid_block_H = min(block_H, kv_group_num)

    shape_q = [batch, heads, dim]
    shape_kv = [total_kv, kv_heads, dim]
    shape_o = [batch, heads, dim]
    part_shape = [batch, heads, num_splits, dim]
    #: 逐 (batch, split) 的 log-sum-exp；q_len = 1 时它是标量
    lse_shape = [batch, num_splits, heads]

    @T.prim_func
    def gqa_decode_splitkv_kernel(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        cu_seqlens_k: T.Tensor([batch + 1], T.int32),
        Output_partial: T.Tensor(part_shape, dtype),
        Lse: T.Tensor(lse_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        # ---------------- split 段 ----------------
        with T.Kernel(batch, heads // valid_block_H, num_splits,
                      threads=threads) as (bid, hid, sid):
            Q_shared = T.alloc_shared([valid_block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_fragment([valid_block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([valid_block_H, block_N], dtype)
            acc_o = T.alloc_fragment([valid_block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([valid_block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([valid_block_H], accum_dtype)
            scores_scale = T.alloc_fragment([valid_block_H], accum_dtype)
            scores_sum = T.alloc_fragment([valid_block_H], accum_dtype)
            logsum = T.alloc_fragment([valid_block_H], accum_dtype)

            # varlen：本序列的 KV 区间由前缀和给出
            cur_start_k = cu_seqlens_k[bid]
            cur_end_k = cu_seqlens_k[bid + 1]
            cur_seqlen_k = cur_end_k - cur_start_k
            # 第 hid 个块的第 0 个 query head 落在哪个 KV head 上
            cur_kv_head = (hid * valid_block_H) // kv_group_num

            T.copy(Q[bid, hid * valid_block_H:
                     hid * valid_block_H + valid_block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # 按比例切：sid 负责 [split_start, split_end)
            split_start = cur_seqlen_k * sid // num_splits
            split_end = cur_seqlen_k * (sid + 1) // num_splits
            split_len = T.ceildiv(split_end - split_start, block_N)
            for k in T.Pipelined(split_len, num_stages=num_stages):
                T.copy(
                    K[cur_start_k + split_start + k * block_N:
                      cur_start_k + split_start + (k + 1) * block_N,
                      cur_kv_head, :],
                    K_shared,
                )
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                # 越界（本 split 之外）的位置记 -inf，不参与 softmax
                for i, j in T.Parallel(valid_block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        split_start + k * block_N + j < split_end,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )
                # online softmax（第 15 章 2.3 的递推，逐项对应）
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(valid_block_H):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(valid_block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale
                                         - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(valid_block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(valid_block_H, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V[cur_start_k + split_start + k * block_N:
                      cur_start_k + split_start + (k + 1) * block_N,
                      cur_kv_head, :],
                    V_shared,
                )
                T.gemm(acc_s_cast, V_shared, acc_o,
                       policy=T.GemmWarpPolicy.FullRow)

            # 写出「本段已归一化的输出」与「本段的 lse」
            for i, j in T.Parallel(valid_block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(valid_block_H):
                # combine 段用 log2 域比较，所以先转成 log2 + 缩放后的最大值
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            for i in T.Parallel(valid_block_H):
                Lse[bid, sid, hid * valid_block_H + i] = logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, hid * valid_block_H:
                                            (hid + 1) * valid_block_H, sid, :])

        # ---------------- combine 段 ----------------
        with T.Kernel(batch, heads // valid_block_H, threads=threads) as (bid, hid):
            po_local = T.alloc_fragment([valid_block_H, dim], dtype)
            o_accum_local = T.alloc_fragment([valid_block_H, dim], accum_dtype)
            o_shared = T.alloc_shared([valid_block_H, dim], dtype)
            lse_local = T.alloc_fragment([num_splits, valid_block_H], dtype)
            lse_local_split = T.alloc_fragment([valid_block_H], accum_dtype)
            lse_logsum_local = T.alloc_fragment([valid_block_H], accum_dtype)
            lse_max_local = T.alloc_fragment([valid_block_H], accum_dtype)
            scale_local = T.alloc_fragment([valid_block_H], accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            for k, j in T.Parallel(num_splits, valid_block_H):
                lse_local[k, j] = Lse[bid, k, hid * valid_block_H + j]
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
            for k in T.serial(num_splits):
                for j in T.Parallel(valid_block_H):
                    # 先减最大值再累加，和 online softmax 是同一个技巧
                    lse_logsum_local[j] += T.exp2(lse_local[k, j]
                                                  - lse_max_local[j])
            for j in T.Parallel(valid_block_H):
                lse_logsum_local[j] = T.log2(lse_logsum_local[j]) + lse_max_local[j]

            for k in T.serial(num_splits):
                T.copy(Output_partial[bid, hid * valid_block_H:
                                      (hid + 1) * valid_block_H, k, :], po_local)
                for j in T.Parallel(valid_block_H):
                    # 权重 = exp(lse_s − logsumexp(lse)) = ℓ_s / Σ_s' ℓ_s'
                    scale_local[j] = T.exp2(lse_local[k, j] - lse_logsum_local[j])
                for i, j in T.Parallel(valid_block_H, dim):
                    o_accum_local[i, j] += po_local[i, j] * scale_local[j]
            T.copy(o_accum_local, o_shared)
            T.copy(o_shared, Output[bid, hid * valid_block_H:
                                    (hid + 1) * valid_block_H, :])

    return gqa_decode_splitkv_kernel


#: 与 `example_mha_inference.py` 的 `num_split = 4` 同一量级；
#: `example_gqa_decode.py` 的 autotune 空间里 num_split ∈ {1, 2, 4, 8}。
DEFAULT_SHAPE = {
    "batch": 8,
    "heads": 28,
    "kv_heads": 4,
    "total_kv": 8 * 8192,
    "max_kv": 8192,
    "dim": 128,
    "num_splits": 8,
}


def build(**overrides: int):
    """编译入口；在有卡的机器上调用它。"""
    cfg = dict(DEFAULT_SHAPE)
    cfg.update(overrides)
    return gqa_decode_splitkv(**cfg)


if __name__ == "__main__":
    raise SystemExit(
        "本文件只做静态核对：请在装有 NVIDIA GPU + TileLang v0.1.15 的机器上\n"
        "参照 code/06-tilelang-decode-attention/README.md 运行。"
    )
