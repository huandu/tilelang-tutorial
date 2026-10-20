"""Autotuning a tiled GEMM in TileLang v0.1.15.

Two entry points are demonstrated, because v0.1.15 ships both:

  * ``@tilelang.autotune(configs=...)`` stacked on ``@tilelang.jit`` — the
    declarative path, used by ``main_tuned``;
  * ``AutoTuner.from_kernel(...).set_compile_args(...).set_profile_args(...)``
    — the programmatic path, used by ``tune_once``.

The config space is produced either from a hand-written Cartesian product or
from Carver's ``MatmulTemplate`` / roller, which recommends device-aware tilings.

Needs an NVIDIA GPU (or an Ascend NPU with the matching dialect) to run. On the
authoring machine only ``python3 -m py_compile`` was run, so the 验证级别 for this
example is 「仅静态核对」. See README.md in this directory.

Prototypes:
  * ``_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_autotune.py``
  * ``_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_advanced_autotune.py``
"""

from __future__ import annotations

import argparse
import itertools
import json

import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


def grid_configs(m: int, n: int, k: int) -> list[dict]:
    """The hand-written Cartesian product, filtered down to legal tiles.

    Two rules are applied *before* the tuner sees the list, because a candidate
    that cannot compile only burns one compile slot:

    * every tile size must divide its problem dimension, so no partial tiles;
    * the tile pair at the requested depth must fit the shared-memory budget
      (see ``../02-shared-memory-budget`` for the arithmetic).
    """
    elem_bytes = 2  # fp16
    # H100 per-SM configurable shared memory ceiling; see ../02-shared-memory-budget.
    smem_budget = 228 * 1024
    configs: list[dict] = []
    for block_m, block_n, block_k, num_stages, threads in itertools.product(
        (64, 128, 256), (64, 128, 256), (16, 32, 64), (1, 2, 3, 4), (128, 256)
    ):
        if m % block_m or n % block_n or k % block_k:
            continue
        need = num_stages * (block_m * block_k + block_n * block_k) * elem_bytes
        if need > smem_budget:
            continue
        configs.append(
            {
                "block_M": block_m,
                "block_N": block_n,
                "block_K": block_k,
                "num_stages": num_stages,
                "threads": threads,
            }
        )
    return configs


def roller_configs(m: int, n: int, k: int, topk: int = 10) -> list[dict]:
    """Carver-generated configs, adapted from the roller's hints.

    Carver ranks tiling strategies with an architecture-aware cost model, then
    hands back ``TileDict`` hints; the mapping from a hint to kernel parameters
    is the caller's job. Imported lazily so that the module can be imported (and
    syntax-checked) without a GPU present.
    """
    from tilelang.carver.arch import CUDA
    from tilelang.carver.roller.rasterization import NoRasterization
    from tilelang.carver.template import MatmulTemplate

    arch = CUDA("cuda")
    carve_template = MatmulTemplate(
        M=m,
        N=n,
        K=k,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float32",
    ).with_arch(arch)

    configs: list[dict] = []
    for hint in carve_template.recommend_hints(topk=topk):
        block_m, block_n = hint.block
        warp_m, warp_n = hint.warp
        rows, cols = block_m // warp_m, block_n // warp_n
        configs.append(
            {
                "block_M": block_m,
                "block_N": block_n,
                "block_K": hint.rstep[0],
                # A hint of 0 or 1 means "no pipeline" in this mapping.
                "num_stages": hint.pipeline_stage if hint.pipeline_stage > 1 else 0,
                "threads": rows * cols * 32,
                "enable_rasterization": hint.rasterization_plan is not NoRasterization,
            }
        )
    return configs


def reference(a, b):
    """Reference program for the correctness check: C = A @ B^T in fp16."""
    return (a.float() @ b.float().T).to(a.dtype)


@tilelang.jit(out_idx=[-1])
def matmul(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """GEMM with the tunable parameters exposed as arguments with defaults."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


# The declarative entry point is the same thing written as a decorator:
#     @tilelang.autotune(configs=..., warmup=..., rep=...)
#     @tilelang.jit(out_idx=[-1])
#     def matmul(...): ...
# ``tilelang.autotune(...)`` returns a decorator that asserts its target is a
# JITImpl, so applying it explicitly is equivalent and lets one factory serve
# both entry points below.
matmul_tuned = tilelang.autotune(
    configs=grid_configs,
    warmup=25,
    rep=100,
    timeout=60,
    ref_prog=reference,
    skip_check=False,
    rtol=1e-2,
    atol=1e-2,
)(matmul)


def tune_once(m: int, n: int, k: int, topk: int) -> None:
    """The programmatic path: inspect the artifact, then bake the winner."""
    from tilelang.autotuner import set_autotune_inputs

    import torch

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)

    tuner = (
        # Note: the *factory*, not the autotuned wrapper above.
        AutoTuner.from_kernel(matmul, configs=roller_configs(m, n, k, topk))
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(
            warmup=25,
            rep=100,
            timeout=60,
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=reference,
            skip_check=False,
        )
    )
    with set_autotune_inputs(a, b, c):
        artifact = tuner.run()

    print("best latency (ms):", artifact.latency)
    print("best config      :", json.dumps(artifact.config, indent=2, sort_keys=True))
    print("reference latency:", artifact.ref_latency)
    artifact.kernel(a, b, c)


def main_tuned(m: int, n: int, k: int) -> None:
    """The declarative path: the decorator tunes on the first call."""
    import torch
    from tilelang.autotuner import set_autotune_inputs

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    with set_autotune_inputs(a, b, c):
        kernel = matmul_tuned(m, n, k)  # compile + tune + return the best kernel
    kernel(a, b, c)
    print("tuned kernel ready; bake the winning parameters as defaults for shipping")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--programmatic",
        action="store_true",
        help="use AutoTuner.from_kernel instead of the decorator",
    )
    args = parser.parse_args()

    if args.programmatic:
        tune_once(args.m, args.n, args.k, args.topk)
    else:
        main_tuned(args.m, args.n, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
