#!/usr/bin/env python3
"""第 18 章 · prefill 与 decode 的算术强度账本（纯标准库，CPU 可跑）。

这份脚本回答三个问题：

1. 同一个模型、同一块卡，为什么 prefill 是 compute-bound 而 decode 是
   memory-bound？
2. 「M 多大才从 memory-bound 翻到 compute-bound」这个临界点怎么算？
   它和 GEMM 的形状有关吗？
3. 一次 decode step 里，「把所有权重读一遍」这件事本身要花多少时间？

口径（与第 02 / 14 / 16 章一致）
-------------------------------

* GEMM ``C[M,N] = A[M,K] @ B[K,N]``：``2*M*N*K`` FLOP；理想情况下
  ``b*(M*K + K*N + M*N)`` 字节。
* 算术强度 ``AI = FLOP / Byte``；与机器平衡点比较即可判类。
* 机器参数取 H100 SXM5 的**公开规格**：fp16 张量核 1000 TFLOPS、
  HBM3 带宽 over 3 TB/s。这两个数字来自厂商公开资料（见
  ``../references.md`` 的 R1/R2），**不是本机测量值**。
* 这是**下界模型**：不含 cache 命中、kernel 启动、同步、通信、尾部效应。
  真实 kernel 一定比这个下界慢；这里只用它判类与找瓶颈。

用法
----

    python3 prefill_decode_roofline.py
    python3 prefill_decode_roofline.py --json
"""

from __future__ import annotations

import argparse
import json

# ---------------------------------------------------------------------------
# 机器参数（厂商公开规格，非实测）
# ---------------------------------------------------------------------------

H100_TFLOPS_FP16 = 1000.0
H100_BW_BYTES_PER_S = 3.0e12
H100_BALANCE = H100_TFLOPS_FP16 * 1e12 / H100_BW_BYTES_PER_S  # 约 333 FLOP/Byte

#: 元素字节数；本章只用到这几种。
ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float8_e4m3": 1}

#: 模型卡给出的字段（`hf-qwen2-5-7b-card`）：28 层、GQA 28 Q / 4 KV、
#: 上下文 131072、参数量 7.61B（非嵌入 6.53B）。
#: ``hidden_size`` / ``head_dim`` / ``intermediate_size`` 模型卡没有直接给，
#: 本章用「参数量必须对上」反推并互校（见 `02-kv-ledger` 的校验函数）。
QWEN25_7B = {
    "n_layers": 28,
    "n_q_heads": 28,
    "n_kv_heads": 4,
    "head_dim": 128,
    "hidden_size": 3584,          # = 28 * 128
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "context": 131072,
    "dtype": "bfloat16",
}


def balance_point(tflops: float, bw_bytes_per_s: float) -> float:
    """机器平衡点 = 峰值算力 / 峰值带宽（FLOP/Byte）。"""
    return tflops * 1e12 / bw_bytes_per_s


def gemm_ai(m: int, n: int, k: int, b: int) -> float:
    """GEMM 的算术强度（上界口径：A、B 各读一次，C 写一次）。"""
    flops = 2.0 * m * n * k
    byts = float(b) * (m * k + k * n + m * n)
    return flops / byts


def verdict(a_intensity: float, balance: float = H100_BALANCE) -> str:
    return "compute-bound" if a_intensity >= balance else "memory-bound"


def crossover_m(n: int, k: int, b: int, balance: float = H100_BALANCE) -> int:
    """二分求最小的 M，使 GEMM 的算术强度达到机器平衡点。"""
    lo, hi = 1, 1
    while gemm_ai(hi, n, k, b) < balance:
        hi *= 2
        if hi > 1 << 30:
            return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if gemm_ai(mid, n, k, b) >= balance:
            hi = mid
        else:
            lo = mid + 1
    return lo


def weight_bytes(cfg: dict[str, int | str]) -> dict[str, float]:
    """按形状算出权重字节数；用来和模型卡的参数量互校。"""
    b = ELEM_BYTES[str(cfg["dtype"])]
    d = int(cfg["hidden_size"])
    i = int(cfg["intermediate_size"])
    h_kv = int(cfg["n_kv_heads"])
    hd = int(cfg["head_dim"])
    v = int(cfg["vocab_size"])
    layers = int(cfg["n_layers"])

    per_layer = d * d + 2 * d * (h_kv * hd) + d * d + 2 * d * i + i * d
    embed = 2 * v * d  # token embedding + lm_head（Qwen2.5-7B 不 tie）
    return {
        "per_layer_params": float(per_layer),
        "non_embedding_params": float(per_layer) * layers,
        "embedding_params": float(embed),
        "total_params": float(per_layer) * layers + embed,
        "total_bytes": (float(per_layer) * layers + embed) * b,
    }


def decode_weight_lower_bound(cfg: dict[str, int | str],
                              bw: float = H100_BW_BYTES_PER_S) -> float:
    """batch=1 时「把权重读一遍」的时间下界（毫秒）。"""
    wb = weight_bytes(cfg)["total_bytes"]
    return wb / bw * 1e3


def print_param_check() -> None:
    wb = weight_bytes(QWEN25_7B)
    print("=" * 78)
    print("先用模型卡的两个参数量反推/互校形状字段")
    print("=" * 78)
    print(f"  非嵌入参数（本书按形状算）: {wb['non_embedding_params'] / 1e9:.3f} B"
          f"   模型卡给出 6.53 B")
    print(f"  总参数（本书按形状算）    : {wb['total_params'] / 1e9:.3f} B"
          f"   模型卡给出 7.61 B")
    print(f"  相对误差                  : "
          f"{abs(wb['total_params'] / 7.61e9 - 1.0):.4%}")
    print("  说明：hidden=3584、head_dim=128、intermediate=18944 三个字段模型卡"
          "\n        没有直接给出，是本章按上面的参数量反推、并与卡片数字互校得到的。")


def print_gemm_crossover() -> None:
    b = ELEM_BYTES[str(QWEN25_7B["dtype"])]
    d = int(QWEN25_7B["hidden_size"])
    i = int(QWEN25_7B["intermediate_size"])
    print("\n" + "=" * 78)
    print(f"同一个 GEMM 形状，M 从 1 涨到 prompt 长度会怎么变（bf16, b={b}）")
    print("=" * 78)

    shapes = [
        ("MLP down-proj  [I -> d]", i, d),
        ("MLP gate/up    [d -> I]", d, i),
        ("QKV 投影       [d -> d+2*Hkv*Hd]", d, d + 2 * 4 * 128),
        ("方形 GEMM（对照）", 4096, 4096),
    ]
    for name, n, k in shapes:
        print(f"\n{name}  N={n} K={k}")
        print(f"{'M':>6} | {'FLOP':>12} | {'bytes':>14} | {'AI (F/B)':>10} | 判定")
        for m in (1, 8, 64, 512, 2048, 8192, 32768):
            ai = gemm_ai(m, n, k, b)
            flops = 2.0 * m * n * k
            byts = float(b) * (m * k + k * n + m * n)
            print(f"{m:>6} | {flops:>12.3e} | {byts:>14.3e} | {ai:>10.1f} | "
                  f"{verdict(ai)}")
        cm = crossover_m(n, k, b)
        print(f"  → 翻转点 M* ≈ {cm}（N、K 固定，只让 M 变）。"
              f"\n    推导：M << N,K 时 AI ≈ 2M/b，与 N、K 无关，于是"
              f"\n    M* ≈ balance * b / 2 = {H100_BALANCE * b / 2:.0f}。"
              f"\n    注意这与第 14 章的方形 GEMM 情形不同：那里 M=N=K 一起涨，"
              f"\n    字节数 ≈ 3b·M²、AI = 2M/(3b)，于是 M* ≈ 3b·balance/2 ≈ 10³。"
              f"\n    两个数都对，前提不同——报 M* 时必须一起报形状。")


def print_attention_ai() -> None:
    """decode attention 的算术强度只取决于「每份 KV 被几个 query head 复用」。"""
    h_q = int(QWEN25_7B["n_q_heads"])
    h_kv = int(QWEN25_7B["n_kv_heads"])
    b = ELEM_BYTES[str(QWEN25_7B["dtype"])]
    print("\n" + "=" * 78)
    print("decode attention：算术强度与序列长度无关（第 16 章的结论）")
    print("=" * 78)
    print("  AI = 2 * H_q / (H_kv * b)")
    print(f"  MHA   (H_kv=28, b={b}) : {2 * h_q / (28 * b):>8.1f} FLOP/Byte")
    print(f"  GQA   (H_kv=4,  b={b}) : {2 * h_q / (h_kv * b):>8.1f} FLOP/Byte")
    print(f"  MQA   (H_kv=1,  b={b}) : {2 * h_q / (1 * b):>8.1f} FLOP/Byte")
    print(f"  机器平衡点             : {H100_BALANCE:>8.1f} FLOP/Byte")
    print("  读法：本模型 GQA 的 KV 复用率是 7（28/4）。这份复用率是 decode"
          "\n        能效的全部来源——换成 MHA 或 MQA/MLA，结论会反过来。")


def print_decode_weight_bound() -> None:
    ms = decode_weight_lower_bound(QWEN25_7B)
    wb = weight_bytes(QWEN25_7B)["total_bytes"]
    print("\n" + "=" * 78)
    print("一次 decode step 的「权重底线」")
    print("=" * 78)
    print(f"  权重字节（bf16）            : {wb / 1e9:.2f} GB")
    print(f"  HBM 带宽（公开规格）        : {H100_BW_BYTES_PER_S / 1e12:.1f} TB/s")
    print(f"  → batch=1 读一遍权重的下界  : {ms:.2f} ms")
    print(f"  → 单请求速度上界（不含 KV） : {1000.0 / ms:.1f} tok/s")
    print("\n  批量会把这份权重流量摊到多个请求上：")
    print(f"{'batch':>6} | {'权重流量/请求(MB)':>18} | {'理想上限(tok/s)':>16}")
    for batch in (1, 2, 8, 32, 128):
        per_req = wb / batch
        print(f"{batch:>6} | {per_req / 1e6:>18.2f} | "
              f"{1000.0 / ms * batch:>16.1f}")
    print("\n  注意：batch 变大时 KV cache 的流量也在涨，上表只算权重，是乐观上界。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args()

    if args.json:
        b = ELEM_BYTES[str(QWEN25_7B["dtype"])]
        payload = {
            "balance_flop_per_byte": H100_BALANCE,
            "params": weight_bytes(QWEN25_7B),
            "decode_weight_lower_bound_ms": decode_weight_lower_bound(QWEN25_7B),
            "gemm_crossover": {
                "mlp_down": crossover_m(int(QWEN25_7B["intermediate_size"]),
                                        int(QWEN25_7B["hidden_size"]), b),
                "square_4096": crossover_m(4096, 4096, b),
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"机器平衡点（H100 SXM5 公开规格 fp16）: {H100_BALANCE:.1f} FLOP/Byte\n")
    print_param_check()
    print_gemm_crossover()
    print_attention_ai()
    print_decode_weight_bound()
    print("\n提醒：以上全部是**下界模型**，用于判类与找瓶颈，不是测量值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
