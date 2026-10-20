#!/usr/bin/env python3
"""第 16 章 · MoE 与 MLA 的成本账本（纯标准库，CPU 可跑）。

这份脚本把第 02 章的 roofline 记账口径套到「大模型一个 decode step」上，
回答四个问题：

1. MoE 的一层里，router / 重排（permute）/ grouped GEMM / combine 各占多少
   FLOP 与多少字节？为什么说瓶颈常常不在矩阵乘？
2. 当一个专家只接到很少的 token 时，这个专家的 GEMM 为什么会退化成
   memory-bound（而这对 MoE 是常态）？
3. MLA 的 KV cache 相对 MHA / GQA 省了多少字节？decode 的算术强度是多少？
4. 一个 decode step 里，哪些算子最贵、分别受什么限制？

记账口径（与第 02 章一致）
-------------------------

* GEMM ``C[M,N] = A[M,K] @ B[K,N]``：``2*M*N*K`` FLOP；最理想情形下
  ``b*(M*K + K*N + M*N)`` 字节（A、B 各读一次，C 写一次），``b`` 为元素字节数。
* 纯搬运算子（permute / gather / scatter）只记字节，不记 FLOP。
* 算术强度 ``AI = FLOP / Byte``；与机器平衡点（第 02 章：H100 SXM5 公开规格下
  约 333 FLOP/Byte 用于 fp16）比较即可判类。
* **这是下界模型**：不含 cache 命中、kernel 启动、同步、通信、尾部效应。
  真实 kernel 一定比这个下界慢。

用法
----

    python3 decode_step_ledger.py            # 打印全部账本
    python3 decode_step_ledger.py --json     # 机器可读输出，便于回归比较
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

#: 锁定版本里 DeepSeek-V3.2-671B 的推理配置；用来把账本落到真实模型形状上。
#: 相对本文件所在目录：01-cost-ledger/ → code/ → 16-operators-moe-mla/
#: → chapters/ → tilelang-book/_vendor/tilelang-v0.1.15/examples/deepseek_v32/inference/
V32_CONFIG_RELATIVE = (
    "../../../../_vendor/tilelang-v0.1.15/examples/"
    "deepseek_v32/inference/config_671B_v3.2.json"
)

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

#: 元素字节数：只覆盖本章用到的 dtype。
ELEM_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3": 1,
    "int8": 1,
}

#: H100 SXM5 的公开规格：fp16 张量核 1000 TFLOPS、HBM3 带宽 over 3 TB/s。
#: 与第 02/14 章同源，只用于「判类」，不是本机测量值。
H100_TFLOPS_FP16 = 1000.0
H100_BW_BYTES_PER_S = 3.0e12
H100_BALANCE = H100_TFLOPS_FP16 * 1e12 / H100_BW_BYTES_PER_S  # 约 333 FLOP/Byte


def b_of(dtype: str) -> int:
    """返回 dtype 的元素字节数。"""
    if dtype not in ELEM_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    return ELEM_BYTES[dtype]


def ai(flops: float, byts: float) -> float:
    """算术强度；字节数为 0 时返回 ``inf``。"""
    if byts <= 0:
        return float("inf")
    return flops / byts


def verdict(a_intensity: float, balance: float = H100_BALANCE) -> str:
    """按机器平衡点判定 compute-bound / memory-bound。"""
    return "compute-bound" if a_intensity >= balance else "memory-bound"


# ---------------------------------------------------------------------------
# 1. MoE 一层的账本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEConfig:
    """一层 MoE 的形状（默认取官方 fusedmoe 示例的量级）。

    默认值来自 ``_vendor/tilelang-v0.1.15/examples/fusedmoe/
    example_fusedmoe_tilelang.py`` 的 ``main()``：d_hidden=7168、d_expert=2048、
    n_routed_experts=8、n_experts_per_token=4。
    """

    num_tokens: int = 8192          # 一层要处理的 token 数（prefill 的一批）
    d_model: int = 7168             # hidden size
    d_expert: int = 2048            # 每个专家的中间维度
    n_routed_experts: int = 8       # 路由专家个数
    top_k: int = 4                  # 每个 token 激活几个专家
    dtype: str = "float16"
    ep_world_size: int = 1          # 专家并行的 rank 数

    def __post_init__(self) -> None:
        if self.top_k > self.n_routed_experts:
            raise ValueError("top_k 不能超过 n_routed_experts")
        if min(self.num_tokens, self.d_model, self.d_expert) <= 0:
            raise ValueError("形状必须为正")

    @property
    def b(self) -> int:
        return b_of(self.dtype)


def moe_layer_math(config: MoEConfig) -> dict[str, float]:
    """MoE 每层的 FLOP / 字节分解（本书模型）。"""
    t, d, de, e, k, b = (
        config.num_tokens,
        config.d_model,
        config.d_expert,
        config.n_routed_experts,
        config.top_k,
        config.b,
    )
    assignments = t * k  # 每层要送进专家的 (token, expert) 对的总数

    # router：logits = x[T, d] @ W_g[d, E]。GEMV 形状，FLOP 极少。
    router_flops = 2.0 * t * d * e
    router_bytes = b * (t * d + d * e + t * e)

    # permute：把 assignments 行 d 维向量按专家重排（读 + 写各一遍）。
    permute_bytes = 2.0 * assignments * d * b

    # grouped GEMM：每个 (token, expert) 对做 gate(d->de) + up(d->de) + down(de->d)。
    grouped_flops = float(assignments) * 6.0 * d * de
    # 专家权重的下界流量：每个被用到的专家读 3 个矩阵。
    weights_bytes = float(e) * 3.0 * d * de * b

    # combine：把 assignments 行结果按 token 聚合（读 + 写各一遍）。
    combine_bytes = 2.0 * assignments * d * b

    # 专家并行的通信：dispatch 送 k 份、combine 收回 k 份，每份 d 个元素。
    comm_bytes = 2.0 * assignments * d * b if config.ep_world_size > 1 else 0.0

    grouped_bytes = weights_bytes + 3.0 * assignments * d * b
    return {
        "assignments": float(assignments),
        "tokens_per_expert_avg": assignments / e,
        "router_flops": router_flops,
        "router_bytes": router_bytes,
        "permute_bytes": permute_bytes,
        "grouped_flops": grouped_flops,
        "weights_bytes": weights_bytes,
        "combine_bytes": combine_bytes,
        "comm_bytes": comm_bytes,
        "reshape_bytes_total": permute_bytes + combine_bytes,
        "grouped_bytes_lower_bound": grouped_bytes,
        "grouped_intensity": ai(grouped_flops, grouped_bytes),
    }


def grouped_gemm_intensity(tokens_for_expert: float, d: int, de: int, b: int) -> float:
    """单个专家自己的 grouped GEMM 的算术强度（本书模型）。

    这个专家单独看是一组 ``M = tokens_for_expert`` 的 GEMM：
    ``2*M*d*de``（gate）+ ``2*M*d*de``（up）+ ``2*M*de*d``（down）FLOP；
    最理想情形下字节 = 权重 ``3*d*de*b`` + 激活 ``O(M*d*b)``。
    """
    if tokens_for_expert <= 0:
        return 0.0
    flops = 6.0 * tokens_for_expert * d * de
    byts = 3.0 * d * de * b + 4.0 * tokens_for_expert * d * b
    return ai(flops, byts)


# ---------------------------------------------------------------------------
# 2. MLA 的账本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLAConfig:
    """MLA 的形状（默认取官方 deepseek_mla 示例）。

    ``deepseek_mla/example_mla_decode.py:218-226`` 的默认参数为
    batch=1、heads=128、kv_heads=1、kv_ctx=8192、dim=512、pe_dim=64。
    官方文档把 MLA 的 head 维度写成 576 = 512 + 64。
    """

    batch: int = 1
    heads: int = 128            # query head 数
    kv_ctx: int = 8192          # KV 序列长度
    dim_c: int = 512            # 压缩潜向量（也是 value）维度
    dim_r: int = 64             # 解耦 RoPE 维度
    dtype: str = "float16"

    @property
    def b(self) -> int:
        return b_of(self.dtype)

    @property
    def dim_k(self) -> int:
        """K（和 Q）的 head 维度：潜向量 + 解耦 RoPE = 576。"""
        return self.dim_c + self.dim_r


def mla_kv_bytes_per_token(cfg: MLAConfig) -> int:
    """MLA 每个 token、每层要缓存的值：c_KV(512) + k_R(64) = 576 个元素。"""
    return (cfg.dim_c + cfg.dim_r) * cfg.b


def mha_like_kv_bytes_per_token(heads_kv: int, head_dim: int, b: int) -> int:
    """MHA / GQA 每个 token、每层要缓存的 K、V 两份。"""
    return 2 * heads_kv * head_dim * b


def mha_like_decode_ai(heads_q: int, heads_kv: int, b: int) -> float:
    """MHA / GQA decode 的算术强度（本书模型）。

    FLOP = ``4*H_q*L*d``（QK 与 PV 各 ``2*H_q*L*d``）；
    字节 = ``2*H_kv*L*d*b``。``L`` 与 ``d`` 都约掉，于是
    ``AI = 2*H_q / (H_kv*b)``——decode 的算术强度与序列长度无关，
    只取决于「每个 KV 元素被几个 query head 复用」。
    """
    if heads_kv <= 0 or b <= 0:
        raise ValueError("heads_kv 与 b 必须为正")
    return 2.0 * heads_q / (heads_kv * b)


def decode_attention(cfg: MLAConfig) -> dict[str, float]:
    """MLA decode（每步 1 个 token）的 FLOP / 字节 / 算术强度（本书模型）。"""
    l, h, b = cfg.kv_ctx, cfg.heads, cfg.b
    dc, dr = cfg.dim_c, cfg.dim_r
    # QK^T 契约维度 = dc + dr；PV 契约维度 = L，输出维度 = dc。
    flops = 2.0 * h * l * (2 * dc + dr)
    byts = float(l) * (dc + dr) * b
    return {"flops": flops, "bytes": byts, "ai": ai(flops, byts)}


# ---------------------------------------------------------------------------
# 3. 打印
# ---------------------------------------------------------------------------


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def print_moe_table() -> None:
    print("=" * 78)
    print("MoE 一层账本（默认形状 = 官方 fusedmoe 示例）")
    print("=" * 78)
    configs = [
        MoEConfig(),
        MoEConfig(num_tokens=128, top_k=6, n_routed_experts=64, d_expert=2048),
        MoEConfig(num_tokens=8192, top_k=6, n_routed_experts=256, d_expert=2048,
                  ep_world_size=16),
    ]
    for cfg in configs:
        m = moe_layer_math(cfg)
        print(
            f"\nT={cfg.num_tokens} d={cfg.d_model} de={cfg.d_expert} "
            f"E={cfg.n_routed_experts} k={cfg.top_k} EP={cfg.ep_world_size}"
        )
        print(f"  assignments (token,expert) 对   : {m['assignments']:.0f}"
              f"  → 平均每专家 {m['tokens_per_expert_avg']:.1f} 个")
        print(f"  router   FLOP / bytes          : {m['router_flops']:.3e} / "
              f"{_fmt_bytes(m['router_bytes'])}")
        print(f"  permute  bytes                 : {_fmt_bytes(m['permute_bytes'])}")
        print(f"  grouped  FLOP                  : {m['grouped_flops']:.3e}")
        print(f"  grouped  bytes（权重下界）     : {_fmt_bytes(m['weights_bytes'])}")
        print(f"  combine  bytes                 : {_fmt_bytes(m['combine_bytes'])}")
        if m["comm_bytes"] > 0:
            print(f"  EP 通信 bytes（dispatch+combine）: {_fmt_bytes(m['comm_bytes'])}")
        print(f"  重排合计（permute+combine）     : {_fmt_bytes(m['reshape_bytes_total'])}")
        print(f"  grouped GEMM 的算术强度        : "
              f"{m['grouped_intensity']:.1f} FLOP/Byte"
              f"  → {verdict(m['grouped_intensity'])}")


def print_expert_small_m() -> None:
    print("\n" + "=" * 78)
    print("为什么「每个专家接到的 token 太少」会让 GEMM 变成 memory-bound")
    print("=" * 78)
    d, de, b = 7168, 2048, 2
    print(f"\n单专家 GEMM：d={d} de={de} fp16（每元素 {b} 字节）")
    print(f"{'tokens_for_expert':>18} | {'FLOP':>12} | {'bytes':>14} | "
          f"{'AI (F/B)':>10} | 判定")
    for tokens in (1, 4, 16, 64, 128, 512, 4096):
        flops = 6.0 * tokens * d * de
        byts = 3.0 * d * de * b + 4.0 * tokens * d * b
        a = ai(flops, byts)
        print(f"{tokens:>18} | {flops:>12.3e} | {_fmt_bytes(byts):>14} | "
              f"{a:>10.1f} | {verdict(a)}")
    print("\n读法：权重（3*d*de*b 约 84 MB）是固定成本。给它 1 个 token，这笔钱"
          "\n也要付一遍；token 数很少时账本几乎全是权重，AI 远低于平衡点。")


def print_mla_table() -> None:
    print("\n" + "=" * 78)
    print("KV cache 与 decode 算术强度")
    print("=" * 78)
    cfg = MLAConfig()
    mla_bytes = mla_kv_bytes_per_token(cfg)
    mha_bytes = mha_like_kv_bytes_per_token(128, 128, cfg.b)
    gqa8_bytes = mha_like_kv_bytes_per_token(8, 128, cfg.b)
    print("\n每个 token、每层的 KV cache（fp16）：")
    print(f"  MHA  128 heads x 128 dim : {mha_bytes:>6} 字节")
    print(f"  GQA  8  kv-heads         : {gqa8_bytes:>6} 字节")
    print(f"  MLA  c_KV(512)+k_R(64)   : {mla_bytes:>6} 字节"
          f"   （= MHA 的 {mla_bytes / mha_bytes:.2%}）")

    print(f"\nMLA decode：batch=1, heads={cfg.heads}, kv_ctx={cfg.kv_ctx}")
    one = decode_attention(cfg)
    print(f"  FLOP  : {one['flops']:.3e}")
    print(f"  bytes : {_fmt_bytes(one['bytes'])}")
    print(f"  AI    : {one['ai']:.1f} FLOP/Byte  → {verdict(one['ai'])}")

    # 对照：MHA/GQA decode 的 AI 只由「每字节 KV 被复用几次」决定。
    mha_ai = mha_like_decode_ai(cfg.heads, 128, cfg.b)
    gqa_ai = mha_like_decode_ai(cfg.heads, 8, cfg.b)
    print(f"\n同为 decode、同为 {cfg.heads} 个 query head 时的算术强度：")
    print(f"  MHA (kv_heads=128, d=128) : {mha_ai:>6.1f} FLOP/Byte")
    print(f"  GQA (kv_heads=8,   d=128) : {gqa_ai:>6.1f} FLOP/Byte")
    print(f"  MLA (kv_heads=1, 576/512) : {one['ai']:>6.1f} FLOP/Byte")


def decode_step_summary() -> list[dict[str, str]]:
    """第 7 节的总结表：一个 decode step 里哪些算子最贵、受什么限制。

    这张表是**本书模型 + 结构化推理**的结论，不是本机测量；每一行的依据都能
    在账本函数与正文引用里找到。
    """
    return [
        {
            "operator": "Router / gating",
            "scale": "T x d_model x E（GEMM 很小）+ top-k",
            "bound_by": "启动延迟与 top-k/排序，基本不吃张量核",
            "notes": "T 是这一步的 token 数；E 很大时 top-k 的开销随之上升",
        },
        {
            "operator": "Token permute / dispatch",
            "scale": "2 x T*k*d_model x b 字节",
            "bound_by": "显存带宽；EP 时是网络 all-to-all",
            "notes": "零 FLOP 的纯搬运，最容易在账本里被忽略",
        },
        {
            "operator": "Grouped GEMM（专家）",
            "scale": "6*T*k*d_model*d_expert FLOP",
            "bound_by": "大 batch 下 compute-bound；小 batch 下被权重流量压成 memory-bound",
            "notes": "每个专家的 M 不齐，且常远小于权重能摊薄的规模",
        },
        {
            "operator": "Combine / unpermute + scatter-reduce",
            "scale": "2 x T*k*d_model x b 字节 + 归约",
            "bound_by": "显存带宽 + 原子/归约的确定性",
            "notes": "k 路求和，浮点加法顺序影响可复现性",
        },
        {
            "operator": "MLA attention（decode）",
            "scale": "T x L x heads x (2*d_c + d_r) FLOP；L x (d_c+d_r) x b 字节/token",
            "bound_by": "L 很大时是 KV 带宽；受益于 128 个 query head 共享 1 份潜向量",
            "notes": "AI 远高于 MHA/GQA decode，因为潜向量被反复复用",
        },
        {
            "operator": "Sparse indexer（仅 DSA/NSA 类）",
            "scale": "约 T x L x d_index FLOP + top-k(L)",
            "bound_by": "与 attention 同量级的访存；top-k 是排序/直方图",
            "notes": "省下的是 attention，付出的是 indexer；两者要一起算",
        },
        {
            "operator": "Dense 投影 / 共享专家 / Norm / RoPE",
            "scale": "O(T x d_model^2) 或 O(T x d_model)",
            "bound_by": "T 很大时 compute-bound（投影）；T=1 时 memory-bound",
            "notes": "decode 单步时是「小而多」的算子，走第 13 章的融合",
        },
    ]


def print_summary_table() -> None:
    print("\n" + "=" * 78)
    print("一个 decode step：哪些算子最贵、分别受什么限制（第 7 节总结表）")
    print("=" * 78)
    rows = decode_step_summary()
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}] {row['operator']}")
        print(f"    规模 : {row['scale']}")
        print(f"    受限 : {row['bound_by']}")
        print(f"    备注 : {row['notes']}")


def load_v32_config() -> dict[str, object]:
    """读锁定版本里的 DeepSeek-V3.2 推理配置；找不到就退回内置常量。

    这个文件在 ``_vendor/tilelang-v0.1.15`` 里，是本章所有「真实模型形状」数字的
    唯一来源（第一方，跟随锁定版本）。它把 MoE 与 MLA 的结构参数给全了：
    ``n_layers=61``、``n_heads=128``、``n_routed_experts=256``、
    ``n_activated_experts=8``、``kv_lora_rank=512``、``qk_rope_head_dim=64``。
    """
    path = Path(__file__).resolve().parent / V32_CONFIG_RELATIVE
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    # 与 config_671B_v3.2.json 一致的兜底值，保证脚本可独立复制运行。
    return {
        "dim": 7168,
        "inter_dim": 18432,
        "moe_inter_dim": 2048,
        "n_layers": 61,
        "n_dense_layers": 3,
        "n_heads": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "dtype": "fp8",
        "index_head_dim": 128,
        "index_topk": 2048,
        "source": "builtin-fallback",
    }


def print_v32_budget() -> None:
    cfg = load_v32_config()
    d = int(cfg["dim"])
    de = int(cfg["moe_inter_dim"])
    layers = int(cfg["n_layers"])
    k = int(cfg["n_activated_experts"])
    e = int(cfg["n_routed_experts"])
    dc = int(cfg["kv_lora_rank"])
    dr = int(cfg["qk_rope_head_dim"])
    src = cfg.get("source", "vendor-config")

    print("\n" + "=" * 78)
    print(f"DeepSeek-V3.2-671B 账本（配置来源：{src}）")
    print("=" * 78)
    print(f"  d_model={d}  d_expert={de}  n_layers={layers}  "
          f"routed={e}  activated={k}")
    print(f"  MLA：kv_lora_rank={dc}  qk_rope_head_dim={dr}  "
          f"缓存每 token/层 = {dc + dr} 个元素")

    per_token_expert = 6.0 * d * de  # 一个专家给一个 token 的全部 FLOP
    per_token_layer = per_token_expert * k
    dense_layers = layers - int(cfg["n_dense_layers"])
    per_token_total = per_token_layer * dense_layers
    print(f"\n  路由专家：每 token 每层 {k} 个专家 x 6*d*de")
    print(f"           = {per_token_expert / 1e6:.1f} MFLOP/专家/token"
          f" → {per_token_layer / 1e6:.1f} MFLOP/token/层")
    print(f"  全模型路由专家（{dense_layers} 层，权重无关的下界）"
          f"：{per_token_total / 1e9:.1f} GFLOP/token")

    kv_per_token_all = (dc + dr) * 2 * layers
    print(f"\n  KV cache：{dc + dr} 元素 x 2 字节 x {layers} 层 "
          f"= {_fmt_bytes(kv_per_token_all)}/token")
    for seq in (8192, 65536, 131072):
        print(f"    序列长度 {seq:>7} 时：{_fmt_bytes(kv_per_token_all * seq)}"
              f"（单条序列，fp16 潜向量）")
    print(f"  对照 MHA(128 heads x 128 dim，含 K/V)："
          f"{_fmt_bytes(2 * 128 * 128 * 2 * layers)}/token")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args()

    if args.json:
        payload = {
            "balance_flop_per_byte": H100_BALANCE,
            "moe_default": moe_layer_math(MoEConfig()),
            "mla_default": {
                "kv_bytes_per_token": mla_kv_bytes_per_token(MLAConfig()),
                "decode": decode_attention(MLAConfig()),
            },
            "deepseek_v32_config": load_v32_config(),
            "decode_step_summary": decode_step_summary(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"机器平衡点（H100 SXM5 公开规格）: {H100_BALANCE:.1f} FLOP/Byte\n")
    print_moe_table()
    print_expert_small_m()
    print_mla_table()
    print_v32_budget()
    print_summary_table()
    print("\n提醒：以上全部是**下界模型**，用于判类与找瓶颈，不是测量值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
