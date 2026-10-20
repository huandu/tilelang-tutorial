#!/usr/bin/env python3
"""第 36 章示例 03：带 skip-scale 的 online softmax（NumPy 参考实现）。

对应正文 §2.2–§2.3 与报告
``FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md:39-73``：

* 状态量：``running_max`` / ``running_max_for_softmax`` / ``running_sumexp`` /
  ``out_accum``；
* 每个 KV 块：gather → ``P = Q·Kᵀ`` → 更新 ``running_max`` → **若滞后超过阈值**
  才 rescale → ``out_accum += S·V``。

⚠️ 这个脚本验证的只是**算法等价性**：分块 + skip-scale 的结果是否与
「一次算完的 softmax」一致。它**不**验证任何硬件行为（FixPipe / MTE2 / icache /
bank conflict），也**不是**对 FlashMLA kernel 的复现。正文 §7 的验证边界同样适用。

验证级别：**本机运行过**（macOS + Python 3.11 + NumPy 1.24）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def attention_reference(
    q: np.ndarray, kv: np.ndarray, indices: np.ndarray, sm_scale: float, dtype
) -> np.ndarray:
    """一次算完的参考实现：gather → P·sm_scale → softmax → PV。"""
    gathered = kv[indices]                                     # [topk, d_qk]
    p = (q.astype(dtype) @ gathered.astype(dtype).T) * sm_scale  # [h_q, topk]
    m = p.max(axis=-1, keepdims=True)
    e = np.exp(p - m)
    s = e / e.sum(axis=-1, keepdims=True)
    return s @ gathered.astype(dtype)


def attention_online_skip_scale(
    q: np.ndarray,
    kv: np.ndarray,
    indices: np.ndarray,
    block: int = 64,
    rescale_thres: float = 6.0,
    sm_scale: float = 1.0,
    dtype=np.float64,
) -> tuple[np.ndarray, int]:
    """分块 + skip-scale 的 online softmax。

    返回 ``(out, num_rescale)``：``num_rescale`` 是实际做了 rescale 的块数。
    注意阈值作用在 ``P * sm_scale`` 上（正文 §2.3），所以比较的是
    缩放后的滞后。
    """
    h_q, d_qk = q.shape
    topk = indices.shape[0]
    d_v = d_qk

    # 四个状态量（报告 :45）
    running_max = np.full(h_q, -np.inf, dtype=dtype)
    running_max_for_softmax = np.full(h_q, -np.inf, dtype=dtype)
    running_sumexp = np.zeros(h_q, dtype=dtype)
    out_accum = np.zeros((h_q, d_v), dtype=dtype)

    num_rescale = 0
    qd = q.astype(dtype)

    for start in range(0, topk, block):
        block_idx = indices[start : start + block]
        gathered = kv[block_idx].astype(dtype)               # [B, d]
        # 报告第 2 步：P = Q @ gathered^T，再乘 sm_scale（阈值也在这个尺度上）
        p = (qd @ gathered.T) * sm_scale                      # [h_q, B]
        cur_max = p.max(axis=-1)                              # [h_q]

        # 报告第 4 步
        running_max = np.maximum(running_max, cur_max)

        # 报告第 5-6 步：只有滞后超过阈值才 rescale
        delta = running_max - running_max_for_softmax
        # 逐行判断：只要有一行触发，就整体 rescale（与 kernel 的
        # "row_max_max_delta > threshold" 口径一致）
        if not np.isfinite(delta).all() or delta.max() > rescale_thres:
            factor = np.exp(running_max_for_softmax - running_max)   # 可能是 -inf -> 0
            factor = np.where(np.isfinite(factor), factor, 0.0)
            out_accum *= factor[:, None]
            running_sumexp *= factor
            running_max_for_softmax = running_max.copy()
            num_rescale += 1

        # 报告第 7 步：S = exp(P - running_max_for_softmax)，累加 S @ V
        with np.errstate(over="ignore", invalid="ignore"):
            s = np.exp(p - running_max_for_softmax[:, None])
        out_accum += s @ gathered

        # 报告第 8 步
        running_sumexp += s.sum(axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = out_accum / running_sumexp[:, None]
    return out, num_rescale


def make_case(
    seed: int, h_q: int, topk: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """正常工况：q、kv ~ N(0,1)，配合 sm_scale = 1/sqrt(d) 后
    ``P·sm_scale`` 的量级是 O(1)（真实 attention 的典型区间）。"""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((h_q, d))
    kv = rng.standard_normal((4 * topk, d))
    indices = rng.choice(kv.shape[0], size=topk, replace=False)
    return q, kv, indices


def make_spiky_case(
    seed: int, h_q: int, topk: int, d: int, spike: float = 40.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """病态工况：后半段 KV 整体放大，使某个后到的块把 logit 顶得很高。

    这是 skip-scale 真正需要防的情形——用它来演示「永不 rescale」在
    float32 下会溢出。
    """
    q, kv, indices = make_case(seed, h_q, topk, d)
    kv = kv.copy()
    half = kv.shape[0] // 2
    kv[half:] *= spike
    return q, kv, indices


def report_run(name: str, out: np.ndarray, ref: np.ndarray, num_rescale: int) -> float:
    abs_err = float(np.max(np.abs(out - ref)))
    denom = np.maximum(np.abs(ref), 1e-30)
    rel_err = float(np.max(np.abs(out - ref) / denom))
    print(f"  {name:<28} rescale={num_rescale:>3d}  max_abs_err={abs_err:.3e}  "
          f"max_rel_err={rel_err:.3e}")
    return abs_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--h-q", type=int, default=64)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--check-torch", action="store_true",
                        help="若装了 torch，再与 torch.softmax 对拍一次")
    args = parser.parse_args()

    q, kv, indices = make_case(args.seed, args.h_q, args.topk, args.d)
    sm_scale = float(args.d) ** -0.5
    ref = attention_reference(q, kv, indices, sm_scale, np.float64)

    print(f"case: seed={args.seed}, h_q={args.h_q}, topk={args.topk}, d={args.d}, "
          f"block={args.block}, sm_scale={sm_scale:.5f}")
    print("比较「分块 online softmax」与「一次算完的参考实现」（float64，正常工况）：")
    num_blocks = (args.topk + args.block - 1) // args.block

    # 逐块、每块都 rescale（阈值 = 0）
    out0, n0 = attention_online_skip_scale(
        q, kv, indices, args.block, rescale_thres=0.0, sm_scale=sm_scale
    )
    report_run(f"rescale_thres=0  (共{num_blocks}块)", out0, ref, n0)

    # 报告里的取值
    out6, n6 = attention_online_skip_scale(
        q, kv, indices, args.block, rescale_thres=6.0, sm_scale=sm_scale
    )
    err6 = report_run("rescale_thres=6  (报告取值)", out6, ref, n6)

    # 永不 rescale（阈值极大）
    outbig, nbig = attention_online_skip_scale(
        q, kv, indices, args.block, rescale_thres=1e30, sm_scale=sm_scale
    )
    report_run("rescale_thres=1e30 (永不)", outbig, ref, nbig)

    print()
    print("病态工况（后半段 KV 放大 40 倍；float32 累加，模拟 kernel 的精度）：")
    qs, kvs, idxs = make_spiky_case(args.seed, args.h_q, args.topk, args.d)
    ref32 = attention_reference(qs, kvs, idxs, sm_scale, np.float32)
    out6s, n6s = attention_online_skip_scale(
        qs, kvs, idxs, args.block, rescale_thres=6.0, sm_scale=sm_scale, dtype=np.float32
    )
    outbs, nbs = attention_online_skip_scale(
        qs, kvs, idxs, args.block, rescale_thres=1e30, sm_scale=sm_scale, dtype=np.float32
    )
    report_run("float32, thres=6", out6s, ref32, n6s)
    bad = not np.isfinite(outbs).all()
    report_run(f"float32, thres=1e30 {'(出现 inf/nan)' if bad else ''}", outbs, ref32, nbs)

    if args.check_torch:
        try:
            import torch
        except ImportError:
            print("  (torch 未安装，跳过)")
        else:
            gathered = kv[indices]
            p = torch.from_numpy(q @ gathered.T).double() * sm_scale
            s = torch.softmax(p, dim=-1)
            torch_ref = (s @ torch.from_numpy(gathered)).numpy()
            print(f"  torch.softmax 与 numpy 参考实现的最大差："
                  f"{float(np.max(np.abs(torch_ref - ref))):.3e}")

    print()
    print("结论：")
    print(f"  1) 正常工况下阈值 = 6 的结果与参考实现一致，且 rescale 次数 "
          f"({n6}/{num_blocks}) 少于每块都做 ({n0}/{num_blocks})；")
    print("  2) 病态工况下「永不 rescale」在 float32 下会溢出（inf/nan），")
    print("     这正是报告 :67-68 给 δ 设上界的动机——阈值不是精度问题，是溢出问题。")
    ok = (err6 < 1e-12) and np.isfinite(out6s).all() and bad
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
