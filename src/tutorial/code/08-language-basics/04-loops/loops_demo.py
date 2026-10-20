"""Chapter 08 / §6: the loop constructs, side by side.

Verification level: **static inspection only** (no GPU / NPU, no TileLang
installed). ``python3 -m py_compile loops_demo.py`` passes. Signatures checked in
``_vendor/tilelang-v0.1.15/tilelang/language/loop.py``:

  * ``Parallel``   :13   (extents, coalesced_width, loop_layout, annotations)
  * ``Pipelined``  :112  (start, stop, num_stages, order, stage, annotations)
  * ``serial``     :189
  * ``unroll``     :230  (explicit, annotations)
  * ``vectorized`` :322
  * aliases ``Serial`` :297 / ``Unroll`` :309 / ``Vectorized`` :349
  * CUDA extras (``T.Parallel(prefer_async=...)``, ``T.unroll(unroll_factor=...)``):
    ``tilelang/cuda/language/loop.py:16-79``
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# serial / vectorized / unroll / Parallel on the same 1-D problem
# ---------------------------------------------------------------------------
@tilelang.jit
def four_loops(N, BLK=128):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, BLK), threads=128) as bx:
            base = bx * BLK

            # T.serial: a plain device loop. One thread walks it alone unless
            # you split the work yourself.
            acc = T.alloc_var("float32", init=0.0)
            for i in T.serial(BLK):
                acc[0] += X[base + i]

            # T.vectorized: the iterations become one vector op (lane count =
            # the extent). Good when the extent matches the useful vector width.
            for i in T.vectorized(4):
                Y[base + i] = X[base + i] + acc[0]

            # T.unroll: ask the backend to emit an unrolled loop. `explicit`
            # forces the unroll in the IR; the CUDA dialect also accepts
            # `unroll_factor=N` (#pragma unroll N).
            for i in T.unroll(4):
                Y[base + 4 + i] = X[base + 4 + i] * 2.0
            for i in T.unroll(4, explicit=True):
                Y[base + 8 + i] = X[base + 8 + i] * 2.0

            # T.Parallel: the compiler owns the thread mapping. extents are the
            # *logical* iteration space; layout inference decides lanes/threads.
            for i in T.Parallel(BLK):
                gi = base + i
                Y[gi] = X[gi]

    return kernel


# ---------------------------------------------------------------------------
# T.Parallel with an explicit layout and with a coalescing hint
# ---------------------------------------------------------------------------
@tilelang.jit
def parallel_hints(M, N, BM=128, BN=128):
    @T.prim_func
    def kernel(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            # coalesced_width= is a hint used by the vectorization planner.
            for i, j in T.Parallel(BM, BN, coalesced_width=4):
                Y[by * BM + i, bx * BN + j] = X[by * BM + i, bx * BN + j] * 2.0

    return kernel


@tilelang.jit
def parallel_custom_layout(M, N, BM=128, BN=128):
    @T.prim_func
    def kernel(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            # `loop_layout` must be a T.Fragment whose InputDim equals the nest
            # depth (here 2). It is attached to the outermost loop only
            # (tilelang/language/loop.py:67-79).
            layout = T.Fragment([BM, BN], forward_thread_fn=lambda i, j: (i * BN + j) % 128)
            for i, j in T.Parallel(BM, BN, loop_layout=layout):
                Y[by * BM + i, bx * BN + j] = X[by * BM + i, bx * BN + j]

    return kernel


# ---------------------------------------------------------------------------
# Pipelined: the copy/compute overlap loop. `num_stages` is the multi-buffer
# depth; `order`/`stage` let you schedule by hand.
# ---------------------------------------------------------------------------
@tilelang.jit
def pipelined_gemm(M, N, K, BM=128, BN=128, BK=32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), "float16")
            B_s = T.alloc_shared((BK, BN), "float16")
            C_f = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_f)
            # num_stages=3 -> up to three A_s/B_s versions in flight.
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])

    return kernel


@tilelang.jit
def pipelined_manual(N, BLK=128):
    @T.prim_func
    def kernel(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, BLK), threads=128) as bx:
            shared = T.alloc_shared((BLK,), "float32")
            # Manual schedule, copied from the T.Pipelined docstring
            # (tilelang/language/loop.py:139-152): two scheduled statements,
            # order=[1, 0] with stage=[0, 1] issues copy-out before copy-in so
            # iteration i's copy-out overlaps iteration i+1's copy-in.
            # The scalar `base` Bind is replayable and consumes no slot.
            for i in T.Pipelined(N // BLK, order=[1, 0], stage=[0, 1]):
                base: T.int32 = i * BLK
                T.copy(A[base], shared)
                T.copy(shared, B[base])

    return kernel


# ---------------------------------------------------------------------------
# while + break/continue inside a device loop
# ---------------------------------------------------------------------------
@tilelang.jit
def while_break(N, BLK=128):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, BLK), threads=128) as bx:
            i = T.alloc_var("int32", init=0)
            while i[0] < BLK:
                if bx * BLK + i[0] >= N:
                    T.loop_break()
                Y[bx * BLK + i[0]] = X[bx * BLK + i[0]]
                i[0] += 1

    return kernel
