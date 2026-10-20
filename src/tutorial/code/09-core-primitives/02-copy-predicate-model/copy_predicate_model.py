#!/usr/bin/env python3
"""模拟 T.copy 的越界谓词与安全值（对应 src/op/copy.cc:449-525）。

    python3 copy_predicate_model.py
    python3 copy_predicate_model.py --safe-value -1.0
    python3 copy_predicate_model.py --dst-size 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Predicate:
    """一个区域检查的结果；generated=False 等价于源码里的空 PrimExpr。"""

    generated: bool
    reason: str


def make_predicate(start: int, extent: int, size: int) -> Predicate:
    """对应 CopyNode::MakePredicate：能证明安全就不生成条件。"""
    if start >= 0 and start + extent <= size:
        return Predicate(True is False, f"{start} + {extent} <= {size}, proved safe")
    return Predicate(True, f"{start} + {extent} > {size} or negative start")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-size", type=int, default=12)
    parser.add_argument("--dst-size", type=int, default=12)
    parser.add_argument("--offset", type=int, default=8)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--safe-value", type=float, default=0.0)
    args = parser.parse_args()

    src = [100.0 + i for i in range(args.src_size)]
    dst = [float("nan")] * args.dst_size

    src_pred = make_predicate(args.offset, args.tile, args.src_size)
    dst_pred = make_predicate(0, args.tile, args.dst_size)

    print(f"src buffer  : {args.src_size} elements, source values = "
          f"[{src[0]:.1f}, {src[1]:.1f}, ..., {src[-1]:.1f}]" if args.src_size >= 2 else "")
    print(f"dst buffer  : {args.dst_size} elements")
    print(f"copy region : src[{args.offset}:{args.offset + args.tile}] -> "
          f"dst[0:{args.tile}]   (tile = {args.tile})")
    print()
    print(f"src predicate: {'generated' if src_pred.generated else 'not generated'} "
          f"({src_pred.reason})")
    print(f"dst predicate: {'generated' if dst_pred.generated else 'not generated'} "
          f"({dst_pred.reason})")
    print()

    print(f"{'lane':>5}  {'src_index':>9}  {'in_bounds':>9}  {'value':>9}  stored")
    guarded = 0
    stored = 0
    for lane in range(args.tile):
        src_index = args.offset + lane
        in_bounds = 0 <= src_index < args.src_size
        if in_bounds:
            value = src[src_index]
        else:
            value = args.safe_value
        if src_pred.generated and not in_bounds:
            guarded += 1
        dst_index = lane
        dst_ok = (not dst_pred.generated) or (0 <= dst_index < args.dst_size)
        if dst_ok:
            dst[dst_index] = value
            stored += 1
        print(
            f"{lane:>5}  {src_index:>9}  "
            f"{('yes' if in_bounds else 'no'):>9}  {value:>9.1f}  "
            f"{'yes' if dst_ok else 'no'}"
        )

    print()
    print(f"guarded src lanes: {guarded} / {args.tile}")
    print(f"stored lanes     : {stored} / {args.tile}")
    print()
    print("pseudo IR:")
    if src_pred.generated:
        print(f"  value = T.if_then_else(src_pred, BufferLoad(A, [{args.offset} + i]), "
              f"{args.safe_value})")
    else:
        print(f"  value = BufferLoad(A, [{args.offset} + i])   # no src guard")
    if dst_pred.generated:
        print(f"  if dst_pred: C[0 + i] = value")
    else:
        print(f"  C[0 + i] = value            # no dst guard needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
