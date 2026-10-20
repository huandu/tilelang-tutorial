#!/usr/bin/env python3
"""第 02 章 示例 1：单个算子的访存/计算账本与 roofline 判据。

这个脚本只回答一个问题：

    给定一个算子和一台机器，它是先撞带宽，还是先撞算力？理论上最快能有多快？

它**不测量任何性能**。所有输出都是公式推导：

    bytes/flops  ->  算术强度 AI = flops / bytes
    sol_time     =  max(bytes / bandwidth, flops / peak_flops)
    判据          =  哪一项决定 sol_time，就是 memory-bound / compute-bound

记账口径（与 TileOPs 的 roofline 模型一致，见
`../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md`）：

* 只数算法**理论最少**的访存：每个输入读一遍、每个输出写一遍，不数中间结果；
* 逐元素的一个基本运算（add/mul/sub/div/max）记 1 FLOP，一个超越函数
  （exp/log/tanh/sqrt/rsqrt/cos/sin…）在账本上也记 1 FLOP；
* 矩阵乘的一次乘加记 2 FLOP。

真实 kernel 一定比这里的下界慢。本脚本给的是「不能被击败」的下界，用来判断
优化方向，而不是预测实现性能。

只用标准库，可在任意机器上离线运行：

    python3 op_traffic.py
    python3 op_traffic.py --machine h100-spec
    python3 op_traffic.py --seq-len 8192 --heads 32 --head-dim 128
    python3 op_traffic.py --params 7e9 --op adamw
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# --------------------------------------------------------------------------
# 机器模型
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Machine:
    """一台机器的两个数：算力峰值与内存带宽。"""

    key: str
    label: str
    peak_flops: float  # FLOP/s，使用张量核 fp16 口径
    bandwidth: float  # Byte/s
    source: str

    @property
    def balance(self) -> float:
        """机器平衡点 = 峰值算力 / 峰值带宽（FLOP/Byte）。"""
        return self.peak_flops / self.bandwidth


# 机器参数一律给出出处；三个 profile 的差别本身就是本章要讲的一件事：
# 「厂商规格」和「能跑到的」不是一个数。
MACHINES: dict[str, Machine] = {
    "h100-spec": Machine(
        key="h100-spec",
        label="H100 SXM5（厂商 preliminary 规格：1000 TFLOPS fp16 / 3.0 TB/s）",
        peak_flops=1000e12,
        bandwidth=3.0e12,
        source=(
            "tilelang-corpus/05-nvidia-backend/raw/"
            "nvidia-blog-hopper-architecture-in-depth.md（Table 1，preliminary specs）"
        ),
    ),
    "h200-spec": Machine(
        key="h200-spec",
        label="H200（厂商规格：989.5 TFLOPS fp16 dense / 4800 GB/s）",
        peak_flops=989.5e12,
        bandwidth=4800e9,
        source=(
            "tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/"
            "profiles/h200.yaml（theoretical 字段，注释写明来自 spec sheet）"
        ),
    ),
    "h200-measured": Machine(
        key="h200-measured",
        label="H200（TileOPs 微基准标定：0.737x 算力 / 0.938x 带宽）",
        peak_flops=989.5e12 * 0.7370,
        bandwidth=4800e9 * 0.9380,
        source=(
            "tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/"
            "profiles/h200.yaml（calibration 字段；算力用 calibration（burst），"
            "带宽用 calibration = max(mixes)，纯 copy 混合只有 0.8183）"
        ),
    ),
}

ELEM_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0}


# --------------------------------------------------------------------------
# 账本
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """一个算子的一次调用的理论最少访存量与运算量。"""

    name: str
    flops: float
    nbytes: float
    assumption: str

    @property
    def intensity(self) -> float:
        return self.flops / self.nbytes


@dataclass(frozen=True)
class Verdict:
    cost: Cost
    memory_time: float
    compute_time: float
    sol_time: float
    bound: str  # "memory-bound" 或 "compute-bound"


def judge(cost: Cost, machine: Machine) -> Verdict:
    memory_time = cost.nbytes / machine.bandwidth
    compute_time = cost.flops / machine.peak_flops
    if memory_time >= compute_time:
        bound = "memory-bound"
    else:
        bound = "compute-bound"
    return Verdict(cost, memory_time, compute_time, max(memory_time, compute_time), bound)


# --------------------------------------------------------------------------
# 各类算子的账本函数
# 每个函数都返回 (flops, bytes) 与一句写在正文里的假设。
# --------------------------------------------------------------------------


def elementwise_add(n: float, dtype: str = "fp16") -> Cost:
    elem = ELEM_BYTES[dtype]
    return Cost(
        name=f"逐元素加 y=x+b  ({n/1e6:.1f}M 元素, {dtype})",
        flops=1.0 * n,
        nbytes=3.0 * n * elem,  # 读 x，读 b，写 y
        assumption="读 x、读 b、写 y；每元素 1 FLOP",
    )


def pointwise_chain(n: float, ops: int, fused: bool, dtype: str = "fp16") -> Cost:
    """一串逐元素算子（PyTorch 文档里的 cos -> sin 就是 2 个）。"""
    elem = ELEM_BYTES[dtype]
    if fused:
        name = f"逐元素链 {ops} 个 op（融合成 1 个 kernel, {dtype}）"
        nbytes = 2.0 * n * elem  # 读 x，写 y
        assumption = "融合后：整条链只读一次输入、写一次输出"
    else:
        name = f"逐元素链 {ops} 个 op（eager, 每个 op 一个 kernel, {dtype}）"
        nbytes = 2.0 * ops * n * elem  # 每个 op 读一次、写一次
        assumption = "eager：每个 op 读一次中间张量、写一次中间张量"
    return Cost(name=name, flops=1.0 * ops * n, nbytes=nbytes, assumption=assumption)


def layernorm(m: float, h: float, fused: bool, dtype: str = "fp16") -> Cost:
    """LayerNorm：M 行、隐藏维 H。中间统计量只有 M 个数，忽略不计。"""
    elem = ELEM_BYTES[dtype]
    per_elem_flops = 8.0  # 均值 1 + 去均值 1 + 平方累加 2 + rsqrt/缩放 2 + 仿射 2
    tensor_elems = m * h
    if fused:
        name = f"LayerNorm 融合（{m:.0f}x{h:.0f}, {dtype}）"
        nbytes = 2.0 * tensor_elems * elem  # 读 x，写 y
        assumption = "读 x、写 y；gamma/beta 与中间统计量忽略不计"
    else:
        name = f"LayerNorm 拆成 3 趟（{m:.0f}x{h:.0f}, {dtype}）"
        nbytes = 4.0 * tensor_elems * elem  # 3 次读 x + 1 次写 y
        assumption = "均值一趟、方差一趟、归一化+仿射一趟：3 次读 x + 1 次写 y"
    return Cost(
        name=name,
        flops=per_elem_flops * tensor_elems,
        nbytes=nbytes,
        assumption=assumption + "；每元素约 8 FLOP（本书口径）",
    )


def gemm(m: float, n: float, k: float, dtype: str = "fp16") -> Cost:
    elem = ELEM_BYTES[dtype]
    return Cost(
        name=f"GEMM {m:.0f}x{n:.0f}x{k:.0f} ({dtype})",
        flops=2.0 * m * n * k,
        nbytes=(m * k + k * n + m * n) * elem,
        assumption="A、B、C 各读/写一遍；一次乘加 2 FLOP",
    )


def attention_prefill(b: float, heads: float, s: float, d: float, materialized: bool,
                      dtype: str = "fp16") -> Cost:
    """prefill（训练前向、推理的 prefill 阶段）的注意力。"""
    elem = ELEM_BYTES[dtype]
    bh = b * heads
    flops = bh * (4.0 * s * s * d + 5.0 * s * s)  # QK + PV 各 2S^2D；softmax 每分数 5
    qkvo_bytes = 4.0 * bh * s * d * elem  # Q、K、V 各读一遍，O 写一遍
    if materialized:
        scores_bytes = 2.0 * bh * s * s * elem  # 写一次 scores
        scores_bytes += 2.0 * bh * s * s * elem  # softmax 读一次、写一次
        scores_bytes += 2.0 * bh * s * s * elem  # PV 再读一次
        name = f"Attention prefill（先落 SxS 分数矩阵, S={s:.0f}）"
        assumption = "scores 写一次、softmax 读+写一次、PV 再读一次（fp16）"
    else:
        scores_bytes = 0.0
        name = f"Attention prefill（flash 风格，不落 SxS, S={s:.0f}）"
        assumption = "分数只在片上存活：Q/K/V/O 各一趟"
    return Cost(
        name=name,
        flops=flops,
        nbytes=qkvo_bytes + scores_bytes,
        assumption=assumption + "；每分数 softmax 记 5 FLOP",
    )


def attention_decode(b: float, heads: float, s: float, d: float, dtype: str = "fp16") -> Cost:
    """decode：每步 1 个 query token，读完整 KV cache。"""
    elem = ELEM_BYTES[dtype]
    bh = b * heads
    flops = bh * (4.0 * s * d + 5.0 * s)
    nbytes = 2.0 * bh * s * d * elem  # 读 KV cache
    return Cost(
        name=f"Attention decode（KV cache 长 {s:.0f}, {dtype}）",
        flops=flops,
        nbytes=nbytes,
        assumption="每个 token 读完整 K/V；q、o 与 KV cache 相比忽略不计",
    )


def adamw(params: float, state_dtype: str = "fp32", grad_dtype: str = "bf16") -> Cost:
    """优化器一步：读写参数、一阶矩 m、二阶矩 v、梯度、权重衰减。"""
    elem_out = ELEM_BYTES[state_dtype]
    elem_grad = ELEM_BYTES[grad_dtype]
    nbytes = (
        elem_grad  # 读 grad
        + 3.0 * elem_out  # 读 param、m、v
        + 3.0 * elem_out  # 写 param、m、v
    ) * params
    return Cost(
        name=f"AdamW step（{params/1e9:.2f}B 参数, state {state_dtype}, grad {grad_dtype}）",
        flops=15.0 * params,
        nbytes=nbytes,
        assumption=(
            f"每参数 {elem_grad:.0f}B grad + {3*elem_out:.0f}B 读 + {3*elem_out:.0f}B 写；"
            "每参数约 15 FLOP（本书口径，按 aclnnApplyAdamW 的公式数）"
        ),
    )


def allreduce(params: float, world: float, grad_dtype: str = "bf16") -> Cost:
    """数据并行梯度 all-reduce（ring）：搬 2x 数据量，且与卡数有关。"""
    elem = ELEM_BYTES[grad_dtype]
    nbytes = 2.0 * params * elem / world * (world - 1.0)
    return Cost(
        name=f"梯度 all-reduce（ring, {world:.0f} 卡, {grad_dtype}）",
        flops=0.0,
        nbytes=nbytes,
        assumption="ring all-reduce 搬 2(N-1)/N 倍梯度数据；没有算术",
    )


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------


def fmt_bytes(value: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value/scale:.2f} {unit}"
    return f"{value:.0f} B"


def fmt_time(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f} s"
    if value >= 1e-3:
        return f"{value*1e3:.3f} ms"
    return f"{value*1e6:.1f} us"


def build_cases(args: argparse.Namespace) -> list[Cost]:
    n_elem = 4096.0 * 4096.0
    cases = [
        elementwise_add(n_elem),
        pointwise_chain(n_elem, ops=2, fused=False),
        pointwise_chain(n_elem, ops=2, fused=True),
        layernorm(m=8192.0, h=4096.0, fused=True),
        layernorm(m=8192.0, h=4096.0, fused=False),
        gemm(4096.0, 4096.0, 4096.0),
        attention_prefill(b=1.0, heads=args.heads, s=args.seq_len, d=args.head_dim,
                          materialized=True),
        attention_prefill(b=1.0, heads=args.heads, s=args.seq_len, d=args.head_dim,
                          materialized=False),
        attention_decode(b=args.batch, heads=args.heads, s=args.kv_len, d=args.head_dim),
        adamw(args.params),
        allreduce(args.params, world=args.world),
    ]
    if args.op != "all":
        wanted = args.op.split(",")
        cases = [c for c in cases if any(w in c.name for w in wanted)]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", default="h200-measured", choices=sorted(MACHINES))
    parser.add_argument("--seq-len", type=float, default=4096.0, help="prefill 序列长度")
    parser.add_argument("--kv-len", type=float, default=32768.0, help="decode 的 KV 长度")
    parser.add_argument("--heads", type=float, default=32.0)
    parser.add_argument("--head-dim", type=float, default=128.0)
    parser.add_argument("--batch", type=float, default=1.0)
    parser.add_argument("--params", type=float, default=7e9, help="优化器参数个数")
    parser.add_argument("--world", type=float, default=8.0, help="数据并行的卡数")
    parser.add_argument("--op", default="all", help="只打印名字里含这些子串的行（逗号分隔）")
    args = parser.parse_args()

    machine = MACHINES[args.machine]
    print("=" * 100)
    print("机器模型")
    print("-" * 100)
    print(f"  profile      : {machine.label}")
    print(f"  峰值算力     : {machine.peak_flops/1e12:.1f} TFLOPS（fp16 张量核口径）")
    print(f"  峰值带宽     : {machine.bandwidth/1e9:.0f} GB/s")
    print(f"  机器平衡点   : {machine.balance:.1f} FLOP/Byte")
    print(f"  来源         : {machine.source}")
    print("=" * 100)
    print("算子账本（理论最少访存量；时间下界 = max(访存时间, 计算时间)）")
    print("-" * 100)
    header = f"{'算子':<50} {'运算量':>10} {'访存量':>10} {'算术强度':>10} {'时间下界':>10} {'判据':>15}"
    print(header)
    for cost in build_cases(args):
        verdict = judge(cost, machine)
        flops = f"{cost.flops/1e9:.3g} GFLOP" if cost.flops > 0 else "0"
        print(
            f"{cost.name:<50} {flops:>10} {fmt_bytes(cost.nbytes):>10} "
            f"{cost.intensity:>9.2f} F/B {fmt_time(verdict.sol_time):>10} {verdict.bound:>15}"
        )
    print("=" * 100)
    print("算术强度 < 平衡点 => 带宽先饱和（memory-bound）；否则算力先饱和（compute-bound）。")
    print("时间下界是不可被击败的下界，真实 kernel 只会更慢；它不是性能预测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
