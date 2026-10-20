#!/usr/bin/env python3
"""第 18 章 · static batching 与 continuous batching 的调度仿真（纯标准库，CPU 可跑）。

这份脚本不测任何真实硬件，它把「调度」这件事单独拎出来做离散事件仿真，
回答一个问题：

    在同样的算力与同样的请求流下，为什么 continuous batching 的吞吐
    能比 static batching 高出一大截？

模型（故意做得非常小，便于手算复核）
----------------------------------

* 一个「槽位（slot）」是一份 KV cache 的容量，也就是一条并发序列。
* 请求按到达时刻进入队列；每个请求有 [(prompt 长度, 输出长度)]。
* 一个 decode step 的耗时固定为 ``STEP_MS``，与这一批里有几条序列无关
  —— 这正是「decode 是 memory-bound、权重流量被 batch 摊薄」在调度层的体现。
* prefill 的耗时正比于本步要处理的 prompt token 数（compute-bound）。

两种策略
--------

* **static**：凑满 ``slots`` 条请求就一起跑到底；早做完的序列空占槽位
  直到整批结束（GPU 在 decode 步里仍然要陪跑）。
* **continuous**：每个 decode step 结束时把已完成的序列踢出去，并从队列
  里补进新序列（这就是 iteration-level scheduling）。

输出：总完成时间、总输出 token 数、吞吐（token/s）、平均与 p95 延迟。

**这是仿真，不是测量**：真实系统里 decode step 的耗时并非严格常数，
prefill 与 decode 还会混步（chunked prefill），调度器还会考虑公平性与
优先级。模型的用处是让「调度决定吞吐」这件事变得可复算。

用法
----

    python3 serving_schedule_sim.py
    python3 serving_schedule_sim.py --slots 8 --step-ms 8 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    prompt: int
    output: int
    arrive_ms: float
    start_ms: float = -1.0
    first_token_ms: float = -1.0
    done_ms: float = -1.0
    produced: int = 0

    @property
    def ttft(self) -> float:
        return self.first_token_ms - self.arrive_ms

    @property
    def e2e(self) -> float:
        return self.done_ms - self.arrive_ms

    @property
    def tpot(self) -> float:
        return (self.done_ms - self.first_token_ms) / max(1, self.output - 1)


@dataclass
class Result:
    name: str
    makespan_ms: float
    tokens: int
    requests: list[Request] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.tokens / (self.makespan_ms / 1e3)

    def latency(self) -> tuple[float, float, float, float]:
        e2e = sorted(r.e2e for r in self.requests)
        ttft = sorted(r.ttft for r in self.requests)
        p95 = e2e[min(len(e2e) - 1, int(0.95 * len(e2e)))]
        return e2e[0], sum(e2e) / len(e2e), p95, sum(ttft) / len(ttft)


def make_requests(n: int, seed: int) -> list[Request]:
    """请求流：输出长度离散度很大，这是真实负载的常态。"""
    rng = random.Random(seed)
    out: list[Request] = []
    t = 0.0
    for i in range(n):
        prompt = rng.choice((128, 256, 512, 1024))
        # 大部分请求几十个 token，少数请求几百个：长尾是 static batching 的杀手
        output = rng.choice((16, 24, 32, 48, 64, 96, 128, 256, 512))
        out.append(Request(i, prompt, output, arrive_ms=t))
        t += rng.uniform(2.0, 12.0)
    return out


def run_static(reqs: list[Request], slots: int, step_ms: float,
               prefill_ms_per_tok: float) -> Result:
    """凑满一批一起跑；早完成的序列空占槽位。"""
    pending = list(reqs)
    clock = 0.0
    done: list[Request] = []
    while pending:
        batch = pending[:slots]
        pending = pending[slots:]
        clock = max(clock, max(r.arrive_ms for r in batch))
        # prefill 阶段：这一批的 prompt 一次性处理
        max_prompt = max(r.prompt for r in batch)
        clock += max_prompt * prefill_ms_per_tok
        for r in batch:
            r.start_ms = clock - max_prompt * prefill_ms_per_tok
            r.first_token_ms = clock
        max_out = max(r.output for r in batch)
        for _ in range(max_out):
            clock += step_ms
            for r in batch:
                if r.produced < r.output:
                    r.produced += 1
                if r.produced == r.output and r.done_ms < 0:
                    r.done_ms = clock
        done.extend(batch)
    return Result("static", clock, sum(r.output for r in done), done)


def run_continuous(reqs: list[Request], slots: int, step_ms: float,
                   prefill_ms_per_tok: float) -> Result:
    """每个 decode step 结束就换人（iteration-level scheduling）。"""
    pending = list(reqs)
    clock = 0.0
    running: list[Request] = []
    done: list[Request] = []

    # 第一步：把队列里能塞进去的请求做 prefill（这里把 prefill 也串在时间轴上）
    while pending or running:
        while pending and len(running) < slots:
            r = pending.pop(0)
            clock = max(clock, r.arrive_ms)
            clock += r.prompt * prefill_ms_per_tok
            r.start_ms = clock - r.prompt * prefill_ms_per_tok
            r.first_token_ms = clock
            r.produced = 1
            if r.produced >= r.output:
                r.done_ms = clock
                done.append(r)
            else:
                running.append(r)
        if not running:
            continue
        # 一个 decode step：所有在跑的序列各出 1 个 token
        clock += step_ms
        still: list[Request] = []
        for r in running:
            r.produced += 1
            if r.produced >= r.output:
                r.done_ms = clock
                done.append(r)
            else:
                still.append(r)
        running = still

    return Result("continuous", clock, sum(r.output for r in done), done)


def print_result(res: Result) -> None:
    p_min, p_mean, p95, ttft_mean = res.latency()
    print(f"\n[{res.name}]")
    print(f"  总耗时 (makespan)      : {res.makespan_ms:>10.1f} ms")
    print(f"  输出 token 总数        : {res.tokens:>10d}")
    print(f"  吞吐                    : {res.throughput:>10.1f} token/s")
    print(f"  E2E 延迟 min/mean/p95  : {p_min:>8.1f} / {p_mean:>8.1f} / "
          f"{p95:>8.1f} ms")
    print(f"  TTFT mean              : {ttft_mean:>10.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--step-ms", type=float, default=8.0,
                        help="一个 decode step 的固定耗时（仿真参数）")
    parser.add_argument("--prefill-us-per-tok", type=float, default=12.0,
                        help="prefill 每个 prompt token 的耗时（微秒）")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    prefill_ms_per_tok = args.prefill_us_per_tok / 1e3
    reqs = make_requests(args.requests, args.seed)

    print("=" * 78)
    print("仿真参数（人为设定，不是测量值）")
    print("=" * 78)
    print(f"  请求数                : {args.requests}")
    print(f"  槽位数（并发上限）    : {args.slots}")
    print(f"  decode step 固定耗时  : {args.step_ms} ms（与 batch 大小无关）")
    print(f"  prefill 每 token      : {args.prefill_us_per_tok} us")
    print(f"  输出长度分布          : {sorted({r.output for r in reqs})}")
    print(f"  输出长度标准差/均值   : "
          f"{_cv([r.output for r in reqs]):.2f}（离散度越大，static 越吃亏）")

    # 两份独立的请求对象：run_* 会就地写 ttft / done_ms，共用对象会互相覆盖
    static = run_static(copy.deepcopy(reqs), args.slots, args.step_ms,
                        prefill_ms_per_tok)
    cont = run_continuous(copy.deepcopy(reqs), args.slots, args.step_ms,
                          prefill_ms_per_tok)
    print_result(static)
    print_result(cont)
    print("\n" + "=" * 78)
    print("对比")
    print("=" * 78)
    speed = static.makespan_ms / cont.makespan_ms
    thr = cont.throughput / static.throughput
    print(f"  总耗时缩短            : {speed:.2f}×")
    print(f"  吞吐提升              : {thr:.2f}×")
    _, sm, sp95, _ = static.latency()
    _, cm, cp95, _ = cont.latency()
    print(f"  平均 E2E 延迟         : {sm:.1f} ms → {cm:.1f} ms")
    print(f"  p95 E2E 延迟          : {sp95:.1f} ms → {cp95:.1f} ms")
    print("\n  读法：收益全部来自「不让早完成的序列占着槽位」和「不让新请求"
          "\n  等到整批结束」。代价是调度器要能处理变长 batch、要按块管理 KV、"
          "\n  还要处理 prefill 与 decode 混步——这就是下一节 varlen 的由来。")
    return 0


def _cv(values: list[int]) -> float:
    """变异系数（标准差 / 均值）。"""
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (var ** 0.5) / mean


if __name__ == "__main__":
    raise SystemExit(main())
