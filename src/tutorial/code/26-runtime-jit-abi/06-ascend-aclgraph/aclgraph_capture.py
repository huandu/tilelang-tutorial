#!/usr/bin/env python3
"""Wrap two TileLang kernels in an Ascend ACLGraph capture.

Chapter 26's launch-overhead section claims that batching many small kernel
submissions into one graph replay is the standard way to escape the
host-bound regime. On Ascend the mechanism is `torch.npu.NPUGraph`, used
exactly like `torch.cuda.CUDAGraph`.

The RMS Norm kernel below is a trimmed copy of the vendor example at
`_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py:21-59`
(capture at line 271); only the second kernel and the shape bookkeeping are
omitted. It is written for an Ascend 950 host; on any other machine it exits
early with a message rather than pretending to run.

API names checked against the locked trees:

  * `T.Kernel(..., is_npu=True)`         vendored example, same file
  * `T.reduce_sum` / `T.tile.*`          vendored example, same file
  * `PassConfigKey.TL_ASCEND_*`          vendored example README
  * `torch.npu.NPUGraph` / `torch.npu.graph`

Usage
-----
    python3 aclgraph_capture.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"{exc.name} is not installed; this example needs an Ascend host.", file=sys.stderr)
        return 2

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("torch.npu is not available on this host.", file=sys.stderr)
        return 2

    import tilelang
    import tilelang.language as T

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }

    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def rms_norm(M: int, head_dim: int, eps: float, block_M: int = 64, dtype: str = "float16"):
        vec_num = 2
        m_num = T.ceildiv(M, block_M)
        row_per_vec = block_M // vec_num
        acc_dtype = "float32"

        @T.prim_func
        def main_rms(
            x: T.Tensor((M, head_dim), dtype),
            out: T.Tensor((M, head_dim), dtype),
        ):
            with T.Kernel(m_num, is_npu=True) as (cid, vid):
                row_x = cid * block_M + vid * row_per_vec
                x_ub = T.alloc_shared([row_per_vec, head_dim], dtype)
                x_ub_fp32 = T.alloc_shared([row_per_vec, head_dim], acc_dtype)
                sum_square_ub = T.alloc_shared([row_per_vec, head_dim], acc_dtype)
                rms_ub = T.alloc_shared([row_per_vec], acc_dtype)

                T.copy(x[row_x : row_x + row_per_vec, :], x_ub)
                T.copy(x_ub, x_ub_fp32)  # fp16 -> fp32
                T.tile.mul(sum_square_ub, x_ub_fp32, x_ub_fp32)
                T.reduce_sum(sum_square_ub, rms_ub, dim=-1)
                T.tile.div(rms_ub, rms_ub, head_dim)
                T.tile.add(rms_ub, rms_ub, eps)
                T.tile.sqrt(rms_ub, rms_ub)
                for i in T.serial(0, row_per_vec):
                    T.tile.div(x_ub_fp32[i, :], x_ub_fp32[i, :], rms_ub[i])
                T.copy(x_ub_fp32, x_ub)  # fp32 -> fp16
                T.copy(x_ub, out[row_x : row_x + row_per_vec, :])

        return main_rms

    M, head_dim = 1024, 512
    kernel = rms_norm(M, head_dim, 1e-6)  # block_M defaults to 64
    x = torch.randn(M, head_dim, device="npu", dtype=torch.float16)
    out = kernel(x)

    # --- graph capture: record, then replay -------------------------------
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = kernel(x)
        sink = captured + out  # two ops inside one capture

    graph.replay()
    torch.npu.synchronize()
    print("replayed shape:", tuple(sink.shape), "dtype:", sink.dtype)

    # Reference check belongs on the same stream, after a synchronisation.
    ref = out + out
    print("matches reference:", bool(torch.allclose(sink, ref, atol=1e-2, rtol=1e-2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
