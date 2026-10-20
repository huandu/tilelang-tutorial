"""Chapter 08 / §3: kernel definition and grid, in TileLang.

Verification level: **static inspection only** (no GPU / NPU on the writing
machine). ``python3 -m py_compile kernel_grid.py`` passes; every API name and
signature below was checked against ``_vendor/tilelang-v0.1.15/``:

  * ``T.Kernel`` (CUDA dialect):
    ``_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-55``
  * grid axis -> ``blockIdx.(x|y|z)``:
    ``_vendor/tilelang-v0.1.15/src/ir.cc:236-262``
  * ``DEFAULT_SIMT_THREADS = 128``:
    ``_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:201``
  * thread accessors:
    ``_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:587-611``
  * Ascend 1-D core grid:
    ``_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-172``

Do NOT run this file expecting output: TileLang is not installed here and there
is no accelerator. It exists so the signatures in the prose can be diffed.
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# 1-D grid: one program per thread block, index bound to bx = blockIdx.x
# ---------------------------------------------------------------------------
@tilelang.jit
def elementwise_scale(N, block=256, dtype="float32"):
    @T.prim_func
    def kernel(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
    ):
        # Positional args of T.Kernel = grid extents, in axis order x, y, z.
        # `threads=` is a launch annotation; omit it and SIMT backends use 128.
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            # Code inside the launch is *block level*, not thread level.
            for i in T.Parallel(block):
                gi = bx * block + i
                Y[gi] = X[gi] * 2.0

    return kernel


# ---------------------------------------------------------------------------
# 2-D grid: two positional extents -> (bx, by) = (blockIdx.x, blockIdx.y)
# ---------------------------------------------------------------------------
@tilelang.jit
def tiled_add(M, N, BM=128, BN=128, dtype="float16"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # First extent -> x (varies fastest), second -> y.
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            for i, j in T.Parallel(BM, BN):
                r = by * BM + i
                c = bx * BN + j
                C[r, c] = A[r, c] + B[r, c]

    return kernel


# ---------------------------------------------------------------------------
# 3-D grid + reading back the launch geometry from inside the body
# ---------------------------------------------------------------------------
@tilelang.jit
def batched_scale(B, M, N, dtype="float32"):
    @T.prim_func
    def kernel(
        X: T.Tensor((B, M, N), dtype),
        Y: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(N, M, B, threads=128) as (bx, by, bz):
            # The same geometry is reachable from inside the body:
            #   T.get_block_binding(dim)  -> the program-index Var for that axis
            #   T.get_block_extent(dim)   -> the grid extent for that axis
            #   T.get_thread_binding(dim) -> threadIdx.(x|y|z)
            # Here thread 0 of each block performs the (scalar) store.
            tid = T.get_thread_binding(0)
            if tid == 0:
                Y[bz, by, bx] = X[bz, by, bx] * 2.0

    return kernel


# ---------------------------------------------------------------------------
# Explicit per-dimension thread shape: threads=(64, 2) -> blockDim=(64,2,1)
# ---------------------------------------------------------------------------
@tilelang.jit
def two_dim_threads(M, N, dtype="float32"):
    @T.prim_func
    def kernel(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, threads=(64, 2)) as bx:
            tx, ty, _tz = T.get_thread_bindings()   # always three (x, y, z)
            Y[bx, ty * 64 + tx] = X[bx, ty * 64 + tx]

    return kernel


# ---------------------------------------------------------------------------
# Ascend: the NPU launch is a 1-D core grid and takes no `threads=`.
# Threads are opened *inside* the body by T.SimtVF.
# ---------------------------------------------------------------------------
def ascend_style_core_grid(NUM_CORES, N, dtype="float32"):
    import tilelang.ascend.language as TA

    @TA.prim_func
    def kernel(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
    ):
        # T.Kernel(NUM_CORES) -> exactly one extent; multi-dim is rejected.
        with TA.Kernel(NUM_CORES) as bx:
            with TA.SimtVF(threads=128):
                for i in TA.Parallel(128):
                    Y[bx * 128 + i] = X[bx * 128 + i] * 2.0

    return kernel
