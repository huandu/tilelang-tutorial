#!/usr/bin/env python3
"""模拟 tl::AllReduce 的 XOR-butterfly（对应 src/tl_templates/cuda/reduce.h:240-295）。

    python3 thread_reduce_model.py --threads 256 --scale 1
    python3 thread_reduce_model.py --threads 48        # 报错，复现 static_assert
"""

from __future__ import annotations

import argparse


class ReduceShapeError(ValueError):
    pass


def validate(threads: int, scale: int) -> None:
    """复现 reduce.h:241-253 的四条 static_assert。"""
    if threads <= 0:
        raise ReduceShapeError("tl::AllReduce threads must be positive")
    if scale <= 0:
        raise ReduceShapeError("tl::AllReduce scale must be positive")
    if threads % scale != 0:
        raise ReduceShapeError("tl::AllReduce threads must be divisible by scale")
    width = threads // scale
    if width & (width - 1) != 0:
        raise ReduceShapeError(
            "AllReduce reduce width (threads / scale) must be a power of two"
        )
    if scale & (scale - 1) != 0:
        raise ReduceShapeError(
            "AllReduce scale (thread stride between reduce participants) must be a power of two"
        )


def butterfly(values: list[float], threads: int, scale: int) -> tuple[list[float], list[int]]:
    """返回 (归约后的值, 每一轮使用的 offset)。"""
    x = list(values)
    offsets: list[int] = []
    offset = threads // 2
    while offset >= scale:
        offsets.append(offset)
        x = [x[t] + x[t ^ offset] for t in range(threads)]
        if offset == scale:
            break
        offset //= 2
    return x, offsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--clear", type=lambda v: v.lower() != "false", default=True)
    parser.add_argument("--out-init", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    try:
        validate(args.threads, args.scale)
    except ReduceShapeError as exc:
        print(f"error: {exc}")
        print("(对应 src/tl_templates/cuda/reduce.h:241-253 的 static_assert)")
        return 2

    # 每个线程贡献自己的那份「线程内归约」结果。
    values = [float(i + 1) for i in range(args.threads)]
    reduced, offsets = butterfly(values, args.threads, args.scale)

    print(
        f"threads = {args.threads}, scale = {args.scale}, "
        f"reduce width = {args.threads // args.scale} -> {len(offsets)} rounds"
    )
    print(f"{'round':>5}  {'offset':>6}  {'mechanism':<9} detail")
    shared_rounds = 0
    for idx, offset in enumerate(offsets, start=1):
        if offset >= 32:
            shared_rounds += 1
            detail = f"red_buf[threadIdx.x] = x; __syncthreads(); x ^= {offset}"
            mech = "shared"
        else:
            detail = f"shfl_xor_sync(-1, x, {offset})"
            mech = "shuffle"
        print(f"{idx:>5}  {offset:>6}  {mech:<9} {detail}")
    print(f"shared-memory rounds: {shared_rounds}")
    print(f"shuffle rounds      : {len(offsets) - shared_rounds}")

    if args.scale == 1:
        value = reduced[0]
    else:
        value = reduced[0]  # 同一组内所有线程得到同一个值
    print(f"reduced value       : {value}")

    if args.clear:
        out = value
        print(f"clear=True  -> out = reduced = {out}")
    else:
        out = args.out_init + value
        print(f"clear=False -> out = out_old + reduced = {args.out_init} + {value} = {out}")

    if args.batch > 1:
        per_thread_outputs = args.threads
        calls = -(-per_thread_outputs // args.batch)
        print(
            f"batch={args.batch} -> {calls} batched AllReduce calls, "
            "each sharing one barrier pair (barrier count / batch)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
