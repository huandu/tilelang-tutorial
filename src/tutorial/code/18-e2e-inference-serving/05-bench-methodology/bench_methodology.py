#!/usr/bin/env python3
"""第 18 章 · 推理基准的口径换算与污染检查（纯标准库，CPU 可跑）。

这份脚本不测性能。它把「报一个推理性能数字之前必须做的事」变成两个
可以机械化的检查：

1. **指标定义**：TTFT / TPOT / ITL / E2E / 吞吐 / 有效吞吐（goodput）的
   定义与它们之间的关系，并演示「同一个数字可以有多少种算法」。
2. **污染检查**：
   * JIT 编译 / autotune / 首次分配落进测量窗口会把数字抬高几个数量级；
   * KV cache、prefix cache、CUDA/ACL graph 都有「第一次慢、之后快」的行为；
   * L2 cache 命中会把带宽类结论抬高；
   * 样本太少时报 p95/p99 没有意义。

用法
----

    python3 bench_methodology.py
    python3 bench_methodology.py --ttft 420 --e2e 9000 --out-tokens 512
"""

from __future__ import annotations

import argparse


def fmt_ms(x: float) -> str:
    return f"{x:,.2f} ms"


def print_definitions() -> None:
    print("=" * 78)
    print("1. 指标定义（缺任何一条，数字就不可比）")
    print("=" * 78)
    rows = [
        ("TTFT", "time to first token",
         "从请求到达到第一个输出 token 生成", "prefill + 排队 + 调度"),
        ("TPOT", "time per output token",
         "(E2E − TTFT) / (输出 token 数 − 1)", "decode 的每一步"),
        ("ITL", "inter-token latency",
         "相邻两个输出 token 的时间间隔分布", "decode 抖动、抢占"),
        ("E2E", "end-to-end latency",
         "从请求到达到最后一个 token", "TTFT + TPOT × (n−1)"),
        ("吞吐", "tokens/s",
         "总输出 token 数 / 墙钟时间", "算上所有并发请求"),
        ("有效吞吐", "goodput",
         "只有满足 SLO（如 TPOT ≤ 50 ms）的请求计入", "服务质量"),
    ]
    print(f"{'指标':<8} | {'全称':<22} | 定义")
    print("-" * 78)
    for abbr, full, definition, note in rows:
        print(f"{abbr:<8} | {full:<22} | {definition}")
        print(f"{'':<8} | {'':<22} |   受什么影响：{note}")


def print_arithmetic(ttft: float, e2e: float, out_tokens: int,
                     in_tokens: int) -> None:
    print("\n" + "=" * 78)
    print("2. 一次请求的换算（同一组数，三种 TPOT 算法）")
    print("=" * 78)
    print(f"  输入 {in_tokens} token，输出 {out_tokens} token")
    print(f"  TTFT = {fmt_ms(ttft)}，E2E = {fmt_ms(e2e)}")
    n = out_tokens
    a = (e2e - ttft) / (n - 1)
    b = (e2e - ttft) / n
    c = e2e / n
    print(f"\n  TPOT = (E2E − TTFT) / (n − 1) = {fmt_ms(a)}   ← 推荐：分母是"
          f"「第一个之后还要出多少个」")
    print(f"  TPOT = (E2E − TTFT) / n       = {fmt_ms(b)}   ← 少数派口径")
    print(f"  E2E / n                       = {fmt_ms(c)}   ← 把 TTFT 摊进每 token，"
          f"不能叫 TPOT")
    print(f"\n  三种算法的差异：{(a - b) / b:.2%} 与 {(c - b) / b:.2%}"
          f"（n={n} 时最多差 1 个 token 的时间 + TTFT/n）")
    print("  结论：**必须写出公式和 n 的口径**（是否含 EOS、是否含第一个 token）。")
    print(f"\n  单请求吞吐 = 1 / TPOT = {1000.0 / a:.1f} tok/s")
    print(f"  单请求「含 TTFT」的稳态速度 = {n / (e2e / 1e3):.1f} tok/s")
    print("  后者在短输出时会明显偏低——这就是短请求的吞吐天然更差的原因。")


def print_pollution(steady_ms: float, first_ms: float, n_calls: int) -> None:
    print("\n" + "=" * 78)
    print("3. 污染检查：首次调用落进测量窗口会怎样")
    print("=" * 78)
    total_dirty = steady_ms * n_calls + first_ms
    total_clean = steady_ms * n_calls
    inflate = total_dirty / total_clean
    print(f"  稳态单次耗时           : {steady_ms:.3f} ms")
    print(f"  首次耗时（含编译/autotune）: {first_ms:,.1f} ms")
    print(f"  测量 {n_calls} 次")
    print(f"  → 含首次：{total_dirty:,.1f} ms（均值 {total_dirty / n_calls:.3f} ms）")
    print(f"  → 去掉首次：{total_clean:,.1f} ms（均值 {steady_ms:.3f} ms）")
    print(f"  → 均值被抬高 {inflate:.2f}×")
    print("\n  这就是为什么 `do_bench` 有 warmup：默认 25 ms 预热 + 100 ms 测量"
          "\n  （`_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31-34`）。")
    print("  注意 warmup 只能吃掉「一次性的」开销：autotune 的搜索、"
          "\n  显存池的首次增长、graph 的 capture，都必须先做一遍再开始计时。")


def print_cache_pollution() -> None:
    print("\n" + "=" * 78)
    print("4. 三类「看起来是真的」的假数字")
    print("=" * 78)
    items = [
        ("KV / prefix cache 命中",
         "TTFT 被压低、吞吐被抬高",
         "请求之间共享 prompt；第二次跑同一批数据",
         "换新 prompt、清 cache，或在报告里注明命中率"),
        ("L2 cache 命中",
         "带宽类指标（GB/s）虚高",
         "kernel 太小，数据整块留在 L2 里",
         "两次调用之间 flush L2；TileLang 的 do_bench 默认 flush 256 MB"
         "（bench.py:42 cache_size=256）"),
        ("CUDA / ACL graph 前后混测",
         "host-bound 的小算子差距被放大或抹平",
         "eager 有 launch 开销，graph 没有",
         "两种模式分别计时，并说明用的是哪种"),
        ("样本太少",
         "p95/p99 没有统计意义",
         "只跑 10~20 次就报 p99",
         "报 p99 至少几百个样本；否则只报 mean/median/min"),
    ]
    for name, symptom, cause, fix in items:
        print(f"\n  ● {name}")
        print(f"      症状：{symptom}")
        print(f"      原因：{cause}")
        print(f"      做法：{fix}")


def print_checklist() -> None:
    print("\n" + "=" * 78)
    print("5. 报告一个推理性能数字前，必须写下来的清单")
    print("=" * 78)
    checks = [
        "硬件型号、卡数、显存（以及是否有 MIG / 多实例切分）",
        "软件版本：推理框架、kernel 库、驱动 / CANN、模型权重版本",
        "模型与精度：权重 dtype、激活 dtype、KV cache dtype、是否量化",
        "并行策略：TP / PP / EP / DP 的度数，是否开启 PD 分离",
        "输入长度分布（均值 + p95）与输出长度（上限 + 采样规则）",
        "并发度：并发请求数、到达率（arrival rate）、是否固定 batch",
        "指标公式：TTFT、TPOT 的分子分母，是否含第一个 token，是否含排队",
        "聚合方式：mean / median / min / p95 / p99，样本数",
        "预热与缓存状态：warmup 次数、是否 flush L2、prefix cache 是否开启",
        "图模式：eager / CUDA graph / ACL graph，以及是否前后混测",
        "对照组的条件是否与本组一致（差异只在被测的那一项）",
        "所有原始日志与脚本可复现（含随机种子）",
    ]
    for i, c in enumerate(checks, 1):
        print(f"  {i:>2}. {c}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttft", type=float, default=420.0)
    parser.add_argument("--e2e", type=float, default=9000.0)
    parser.add_argument("--out-tokens", type=int, default=512)
    parser.add_argument("--in-tokens", type=int, default=1024)
    parser.add_argument("--steady-ms", type=float, default=0.35)
    parser.add_argument("--first-ms", type=float, default=3200.0)
    parser.add_argument("--n-calls", type=int, default=100)
    args = parser.parse_args()

    print_definitions()
    print_arithmetic(args.ttft, args.e2e, args.out_tokens, args.in_tokens)
    print_pollution(args.steady_ms, args.first_ms, args.n_calls)
    print_cache_pollution()
    print_checklist()
    print("\n提醒：本脚本不测量任何性能，只做定义与检查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
