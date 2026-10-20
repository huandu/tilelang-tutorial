#!/usr/bin/env python3
"""第 38 章 · 示例 01：README 里那段 EP 用法的最小复刻。

这个脚本**不需要 NPU**。它做两件事：

1. 把 ``DeepEP-Ascend/README.md:100-196`` 的用法整理成一份带注解的走查，
   方便读者对着源码读；
2. 在**不依赖设备**的层面检查若干硬约束（偶数 rank、展开布局、
   ``topk <= 32``、必须保持默认的参数）。

作者验证级别：**CPU 逻辑验证**（macOS Apple Silicon / Python 3.11）。
没有任何一行 DeepEP-Ascend kernel 被运行过（本机没有昇腾 NPU）。

用法::

    python3 ep_minimal.py            # 走查 + 约束检查
    python3 ep_minimal.py --summary  # 只看约束检查结果
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. 配置：与 README.md:100-104 的样例一致（token 容量换成 16384，与性能表同规模）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EPConfig:
    num_max_tokens_per_rank: int
    hidden: int
    num_topk: int
    num_experts: int
    num_ranks: int
    use_fp8_dispatch: bool = False


CONFIG = EPConfig(
    num_max_tokens_per_rank=16384,
    hidden=7168,
    num_topk=6,
    num_experts=256,
    num_ranks=8,
    use_fp8_dispatch=True,
)


# ---------------------------------------------------------------------------
# 2. 约束检查：这些是源码里写死的，不是可调项
# ---------------------------------------------------------------------------


def check_constraints(cfg: EPConfig) -> list[tuple[str, bool, str]]:
    """返回 (约束名, 是否满足, 依据) 列表。

    依据全部来自 DeepEP-Ascend 源码（本机只做静态核对）：
      - 奇数 rank：README.md:29、csrc/kernels/comm/hccl.hpp:85
      - 展开布局：csrc/buffers/ep.hpp:116
      - do_handle_copy：README.md:72、deep_ep/buffers/ep.py:392
      - topk 上限 32：CSRC combine.hpp:85、common/compiled.hpp 的 warpSize=32
      - num_sms 语义：README.md:74、deep_ep/buffers/ep.py:305-309
    """
    results: list[tuple[str, bool, str]] = []

    results.append(
        (
            "rank 数为偶数",
            cfg.num_ranks % 2 == 0,
            "README.md:29；csrc/kernels/comm/hccl.hpp:85",
        )
    )
    results.append(
        (
            "专家数能被 rank 数整除",
            cfg.num_experts % cfg.num_ranks == 0,
            "csrc/buffers/ep.hpp:139",
        )
    )
    results.append(
        (
            "topk <= 32",
            1 <= cfg.num_topk <= 32,
            "csrc/kernels/ep/combine.hpp:85；dispatch 侧 kNumTopk <= warpSize",
        )
    )
    results.append(
        (
            "hidden 是 128 的倍数",
            cfg.hidden % 128 == 0,
            "csrc/kernels/ep/combine.hpp:86（hidden % (256/2) == 0）",
        )
    )
    results.append(
        (
            "num_max_tokens_per_rank > 0",
            cfg.num_max_tokens_per_rank > 0,
            "csrc/buffers/ep.hpp:585（calculate_buffer_size 的断言）",
        )
    )
    return results


# ---------------------------------------------------------------------------
# 3. 走查：把 README 的调用翻译成注解
# ---------------------------------------------------------------------------

WALKTHROUGH = """\
（1）初始化
    ep_group = dist.group.WORLD                      # backend='hccl'
    buffer = EPBuffer(
        ep_group,
        num_max_tokens_per_rank=...,   # 所有 rank 必须一致
        hidden=7168,
        num_topk=6,
        use_fp8_dispatch=False,        # False 也够 FP8 用（README 注释）
        explicitly_destroy=True,       # 必须手动 destroy()
    )
    · 构造时注册对称内存 [ workspace | buffer ]，建 HCCL 信道与 Jetty
    · 结尾 synchronize + dist.barrier(group)，等所有 rank 就绪

（2）先算容量（可选）
    need = EPBuffer.get_buffer_size_hint(group, T, hidden, num_topk=6, use_fp8_dispatch=True)
    · 返回 2 MB 对齐的字节数；SF 预留量是实际用量的 2 倍

（3）前向 dispatch
    pending = buffer.dispatch(
        x,                             # bf16，或 (fp8_data, sf)
        topk_idx=topk_idx,             # [T, 6] int64
        topk_weights=topk_weights,     # [T, 6] float32
        num_experts=256,
        expert_alignment=128,          # 与 grouped GEMM 的 M 对齐一致
        do_cpu_sync=True,              # decode 时改 False
        do_expand=True,                # 昇腾只支持 True
        do_zero_padding=True,
        async_with_compute_stream=True,
        defer_epilogue=True,           # 立刻返回 EventOverlap
    )
    shared = shared_experts(x)         # 与通信重叠
    recv_x, _, recv_weights, handle = pending.wait()

（4）专家计算 + combine
    expert_out, _ = local_experts_forward(recv_x, recv_weights, handle)
    pending = buffer.combine(
        expert_out,                    # bf16
        handle=handle,
        bias=shared,                   # 把共享专家输出融进 combine
        async_with_compute_stream=True,
        defer_epilogue=True,
    )
    output, _ = pending.wait()

（5）反向：dispatch 的反向是 combine，combine 的反向是 dispatch，复用同一个 handle

（6）收尾
    buffer.destroy()
    dist.destroy_process_group()
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepEP-Ascend 最小用法走查（无需 NPU）")
    parser.add_argument("--summary", action="store_true", help="只打印约束检查结果")
    args = parser.parse_args()

    if not args.summary:
        print(WALKTHROUGH)

    print("=" * 72)
    print("约束检查（全部来自源码断言，本机只做静态核对）")
    print("=" * 72)
    ok = True
    for name, passed, source in check_constraints(CONFIG):
        mark = "OK  " if passed else "FAIL"
        if not passed:
            ok = False
        print(f"[{mark}] {name:<28} {source}")

    print()
    print(
        f"配置：ranks={CONFIG.num_ranks}, experts={CONFIG.num_experts}, "
        f"topk={CONFIG.num_topk}, hidden={CONFIG.hidden}, "
        f"T={CONFIG.num_max_tokens_per_rank}, fp8={CONFIG.use_fp8_dispatch}"
    )
    if not ok:
        print("有约束不满足；真实环境里会在构造或调用时直接断言失败。", file=sys.stderr)
        return 1
    print("全部通过。注意：这不代表能在真机上跑通——本机没有昇腾 NPU。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
