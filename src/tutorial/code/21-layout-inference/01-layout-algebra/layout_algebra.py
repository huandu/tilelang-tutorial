#!/usr/bin/env python3
"""第 21 章 §2 的 layout 代数复算（纯标准库，不需要 TileLang）。

复算四件事：
  1. 转置映射是否双射；
  2. padding 映射的物理空间与空洞（非双射）；
  3. 一个丢位的映射为什么会让 inverse 的 round-trip 检查失败；
  4. MakeGemmFragment8x8 的线程/槽位分配是否无碰撞。

参考（v0.1.15）：
  _vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:45-52   MakeGemmFragment8x8
  _vendor/tilelang-v0.1.15/src/layout/layout.cc:1150-1160    inverse 的 round-trip 检查
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product


def classify(mapping, in_shape):
    """给一个映射统计：值域大小（是否满射）、是否有碰撞（是否单射）。"""
    counts: dict = defaultdict(list)
    for idx in product(*(range(s) for s in in_shape)):
        counts[mapping(*idx)].append(idx)
    collisions = sum(len(v) - 1 for v in counts.values())
    return counts, collisions


def transpose(i: int, j: int):
    return (j, i)


def padded(i: int, j: int):
    return i * 10 + j


def broken(i: int, j: int):
    """故意丢掉低位：把 2x2 的块压成一个点，构造非单射映射。"""
    return (i // 2) * 4 + (j // 2)


def gemm_fragment_8x8(i: int, j: int):
    """复刻 MakeGemmFragment8x8：(i, j) -> (thread, slot)。"""
    thread = j // 2 + 4 * i
    slot = j % 2
    return (thread, slot)


def main() -> None:
    # 1) 转置：双射
    counts, collisions = classify(transpose, (8, 8))
    is_bijective = collisions == 0 and len(counts) == 8 * 8
    print(f"transpose [8,8]: bijective={is_bijective}, points={len(counts)}")

    # 2) padding：非满射
    counts_p, collisions_p = classify(padded, (8, 8))
    used = set(counts_p)
    holes = sorted(set(range(80)) - used)
    print(
        f"padded [8,8] -> stride 10: used={len(used)}, holes={len(holes)}, "
        f"is_surjective={len(used) == 80}, collisions={collisions_p}"
    )
    assert collisions_p == 0, "padding 不应产生碰撞"
    assert len(holes) == 16, "8 行 x 2 个空槽 = 16"

    # 3) 丢位映射：有碰撞且往返不一致
    counts_b, collisions_b = classify(broken, (8, 8))
    mismatch = 0
    for i, j in product(range(8), range(8)):
        out = broken(i, j)
        # 取一个能映到 out 的代表点，比较是否回到原点
        representative = next(pt for pt in product(range(8), range(8))
                              if broken(*pt) == out)
        if representative != (i, j):
            mismatch += 1
    print(f"broken map: collisions={collisions_b}, round-trip mismatch={mismatch}")
    assert collisions_b > 0 and mismatch > 0

    # 4) MMA 8x8 store fragment：无碰撞，32 线程、每线程 2 槽
    counts_f, collisions_f = classify(gemm_fragment_8x8, (8, 8))
    threads = {t for t, _s in counts_f}
    per_thread_slots = defaultdict(set)
    for (t, s) in counts_f:
        per_thread_slots[t].add(s)
    slot_counts = {len(v) for v in per_thread_slots.values()}
    print(
        f"MakeGemmFragment8x8: threads={len(threads)}, "
        f"slots_per_thread={slot_counts}, bijective={collisions_f == 0}"
    )
    assert collisions_f == 0
    assert len(threads) == 32
    assert slot_counts == {2}


if __name__ == "__main__":
    main()
