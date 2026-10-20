"""Chapter 08 / §5: the allocation functions and the scope each one declares.

Verification level: **static inspection only** (no GPU / NPU, no TileLang
installed). ``python3 -m py_compile alloc_scopes.py`` passes. Every signature
below was read out of ``_vendor/tilelang-v0.1.15/``:

  * shared / local / fragment / var / global:
    ``tilelang/language/allocate.py:49, 67, 81, 103, 171``
  * the CUDA bool workaround that rewrites the scope to "shared":
    ``tilelang/language/allocate.py:60-62``
  * Blackwell Tensor Memory:
    ``tilelang/language/allocate.py:254-289``
  * Ascend UB / L1 / L0A / L0B / L0C:
    ``tilelang/ascend/language/allocate.py:15, 26, 37, 46, 55``
  * the adapter repo spells these ``alloc_ub / alloc_L1 / alloc_L0A / ...``:
    ``_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:140-157``
    (note: ``alloc_ub`` does NOT exist in mainline v0.1.15)
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# One kernel that shows all four "general" scopes side by side.
# ---------------------------------------------------------------------------
@tilelang.jit
def scope_showcase(M, N, K, BM=128, BN=128, BK=32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            # shared.dyn: block-visible scratch, the staging area for tiles.
            A_s = T.alloc_shared((BM, BK), "float16")
            B_s = T.alloc_shared((BK, BN), "float16")
            # local.fragment: the logical accumulator. It *looks* like a
            # block-shaped buffer, but LayoutInference splits it across the
            # block's registers, so its real footprint is per-thread.
            C_f = T.alloc_fragment((BM, BN), "float32")
            # local: a per-thread scratch array with an ordinary index space.
            scratch = T.alloc_local((BK,), "float32")
            # local.var: a single scalar. `init=` seeds it; without `init=`
            # it starts at the dtype's zero.
            scale = T.alloc_var("float32", init=0.5)

            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)

            for i, j in T.Parallel(BM, BN):
                C_f[i, j] = C_f[i, j] * scale[0]
            for k in T.serial(BK):
                scratch[k] = 0.0
            T.copy(C_f, C[by * BM, bx * BN])

    return kernel


# ---------------------------------------------------------------------------
# alloc_global is a workspace allocation (cudaMalloc-style), not a kernel
# argument. The vendor docstring recommends passing your own buffer instead.
# ---------------------------------------------------------------------------
def global_workspace(N):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            ws = T.alloc_global((128,), "float32")  # allocated by the backend
            for i in T.Parallel(128):
                ws[i] = X[bx * 128 + i]
                Y[bx * 128 + i] = ws[i]

    return kernel


# ---------------------------------------------------------------------------
# Blackwell: Tensor Memory is its own scope ("shared.tmem") and has its own
# rules (power-of-two column count >= 32, allocate/deallocate by one warp).
# ---------------------------------------------------------------------------
def tmem_accumulator(BM, BN):
    @T.prim_func
    def kernel(A: T.Tensor((BM, BN), "float16"), C: T.Tensor((BM, BN), "float32")):
        with T.Kernel(1, threads=128):
            # scope "shared.tmem"; column count must be a power of two and >= 32.
            # TileLang frees it at the end of the block unless you call
            # T.deallocate_tmem yourself.
            # Only TCGEN5.MMA and the TMEM load/store instructions may touch
            # this buffer; that lowering is driven by T.gemm on Blackwell.
            # The explicit `tcgen05_gemm` entry point lives in
            # tilelang/cuda/language/gemm_op.py and is NOT re-exported by
            # `import tilelang.language as T` in v0.1.15.
            C_t = T.alloc_tmem((BM, BN), "float32")
            A_s = T.alloc_shared((BM, BN), "float16")
            T.copy(A, A_s)

    return kernel


# ---------------------------------------------------------------------------
# Ascend: UB is `T.alloc_shared`; L1 / L0A / L0B / L0C have their own names.
# ---------------------------------------------------------------------------
def ascend_scope_showcase(M, N, K, BM=256, BN=224, BK=128):
    import tilelang.ascend.language as TA

    @TA.prim_func
    def kernel(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with TA.Kernel(32) as bx:
            A_l1 = TA.alloc_l1((BM, BK), "bfloat16")       # scope shared.l1
            B_l1 = TA.alloc_l1((BN, BK), "bfloat16")       # scope shared.l1
            C_l0c = TA.alloc_l0c((BM, BN), "float32")      # scope shared.l0c
            C_ub = TA.alloc_shared((BM // 2, BN), "float32")  # Unified Buffer
            # alloc_l0a / alloc_l0b exist too; layout is deferred to the
            # consuming T.gemm, which infers K-major vs MN-major from its
            # transpose flags (`tilelang/ascend/language/allocate.py:37-52`).
            TA.clear(C_ub)
            TA.clear(C_l0c)
            TA.copy(A[bx * BM, 0], A_l1)
            TA.copy(B[0, 0], B_l1)
            TA.gemm(A_l1, B_l1, C_l0c, transpose_B=True)
            TA.copy(C_ub, C[bx * BM, 0])

    return kernel


# ---------------------------------------------------------------------------
# The adapter repo spells the same five buffers differently. Keep both spellings
# straight: the scope names differ too, so a snippet copied between the two
# trees will not necessarily drop in.
# ---------------------------------------------------------------------------
ADAPTER_SPELLINGS = {
    # adapter repo (`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py`)
    "T.alloc_shared(shape, dtype)": "scope 'shared'  (bool -> 'shared.ub')",
    "T.alloc_ub(shape, dtype)": "scope 'shared.ub'",
    "T.alloc_L1(shape, dtype)": "scope 'shared.l1'",
    "T.alloc_L0A(shape, dtype)": "scope 'wmma.matrix_a'",
    "T.alloc_L0B(shape, dtype)": "scope 'wmma.matrix_b'",
    "T.alloc_L0C(shape, dtype)": "scope 'wmma.accumulator'",
    # mainline v0.1.15 (`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py`)
    "T.alloc_shared(...)  (ascend dialect)": "scope 'shared.dyn'  = Unified Buffer",
    "T.alloc_l1(shape, dtype)": "scope 'shared.l1'",
    "T.alloc_l0a(shape, dtype)": "scope 'shared.l0a'",
    "T.alloc_l0b(shape, dtype)": "scope 'shared.l0b'",
    "T.alloc_l0c(shape, dtype)": "scope 'shared.l0c'",
}
