"""Chapter 08 / §7: control flow inside a kernel, and what it costs.

Verification level: **static inspection only** (no GPU / NPU, no TileLang
installed). ``python3 -m py_compile control_flow_demo.py`` passes.

Sources checked in ``_vendor/tilelang-v0.1.15/``:

  * ``T.if_then_else`` / ``T.Select`` / ``T.min`` / ``T.max``:
    ``tilelang/language/tir/ir.py:172-186, 228``
    ("Select evaluates both value expressions. Use if_then_else when a branch
     must guard side effects or out-of-bounds memory accesses.")
  * ``T.any_of`` / ``T.all_of``: ``tilelang/language/logical.py:12, 51``
    -- these take a **buffer or buffer region**, not a list of conditions.
  * ``and`` / ``or`` / ``not`` on TIR expressions:
    ``tilelang/language/eager/builder.py:820-846`` (lowered to And/Or/Not).
  * ``T.loop_break``: ``tilelang/language/customize.py:163``
  * ``T.print``: ``tilelang/language/builtin.py`` (single-thread print)
  * proxies for the semantics: ``docs/programming_guides/control_flow.md``
    (that file's ``T.all_of(i < M, j < N)`` example does NOT match
    ``logical.py`` -- see the note in this file and in §9 of the chapter).
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# 1. if / elif / else, with a TIR condition, and a ternary.
# ---------------------------------------------------------------------------
@tilelang.jit
def branching(M, N, BLK=128):
    @T.prim_func
    def kernel(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((M, N), "float32"),
        S: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLK), M, threads=128) as (bx, by):
            # `tid` is a Python int here: this `if` is decided at trace time and
            # simply disappears from the generated code.
            tid = 0
            if tid == 0:
                pass

            for i in T.Parallel(BLK):
                gi = bx * BLK + i
                # TIR condition -> a real device-side branch.
                if gi < N:
                    Y[by, gi] = X[by, gi]
                elif gi == N:
                    Y[by, gi] = 0.0
                else:
                    Y[by, gi] = -1.0

            # `x if cond else y` builds T.if_then_else, i.e. a value, not a branch.
            Y[by, 0] = S[by] if by < M else 0.0

    return kernel


# ---------------------------------------------------------------------------
# 2. Statement `if` vs value `if_then_else` vs value `Select`.
#    The three are not interchangeable.
# ---------------------------------------------------------------------------
@tilelang.jit
def safe_load_vs_select(N, BLK=128):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, BLK), threads=128) as bx:
            for i in T.Parallel(BLK):
                gi = bx * BLK + i

                # SAFE: if_then_else only evaluates the branch it needs, so the
                # out-of-range X[gi] load never happens when the guard is false.
                Y[gi] = T.if_then_else(gi < N, X[gi], 0.0)

                # UNSAFE as written: Select evaluates *both* operands, so X[gi]
                # is materialised even when gi >= N. Use it only when both
                # operands are safe to touch.
                #   Y[gi] = T.Select(gi < N, X[gi], 0.0)

    return kernel


# ---------------------------------------------------------------------------
# 3. any_of / all_of take a buffer (or a buffer slice), not a condition list.
# ---------------------------------------------------------------------------
@tilelang.jit
def buffer_predicates(BLK=128, KV=64):
    @T.prim_func
    def kernel(
        Mask: T.Tensor((BLK, KV), "uint8"),
        Flag: T.Tensor((BLK,), "uint8"),
    ):
        with T.Kernel(1, threads=128) as bx:
            mask_local = T.alloc_shared((KV,), "uint8")
            T.copy(Mask[bx, 0], mask_local)
            # all_of(buffer) -> "every element of this slice is true"
            Flag[bx] = T.if_then_else(T.all_of(mask_local), T.uint8(1), T.uint8(0))
            # any_of(buffer) -> "at least one element of this slice is true"
            Flag[bx] = T.if_then_else(T.any_of(mask_local[0:KV]), T.uint8(1), Flag[bx])
            # For a *condition list* use `and` / `or`, which lower to And/Or:
            #   flag = T.if_then_else((bx < BLK) and (KV > 0), 1, 0)

    return kernel


# ---------------------------------------------------------------------------
# 4. while, with break / continue.
# ---------------------------------------------------------------------------
@tilelang.jit
def while_scan(N, BLK=64):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=128) as bx:
            i = T.alloc_var("int32", init=0)
            # The condition must be a TIR expression; a constant-true condition
            # is rejected by the compiler.
            while i[0] < BLK:
                if i[0] >= N:
                    T.loop_break()
                if X[i[0]] < 0.0:
                    i[0] += 1
                    continue
                Y[i[0]] = X[i[0]]
                i[0] += 1

    return kernel


# ---------------------------------------------------------------------------
# 5. A guard that cannot be hoisted blocks vectorization; a pre-split loop
#    keeps the fast path vectorizable.
# ---------------------------------------------------------------------------
@tilelang.jit
def guard_costs_vectorization(N, threads=256, num_per_thread=8):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as bx:
            # Every iteration carries `gi < N`, so the planner cannot emit a
            # straight vector load -- it emits a predicated scalar loop.
            for i, j in T.Parallel(threads, num_per_thread):
                gi = (bx * threads + i) * num_per_thread + j
                if gi < N:
                    Y[gi] = X[gi] * 2.0

    return kernel


@tilelang.jit
def guard_free_full_blocks(N, threads=256, num_per_thread=8):
    """Cover only the full blocks, so the fast path needs no guard at all."""
    BLK = threads * num_per_thread
    full = (N // BLK) * BLK

    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        # Grid covers `full` elements only: every gi is provably < N, so both
        # the guard and the tail disappear -- the planner is free to vectorize.
        with T.Kernel(T.ceildiv(full, BLK), threads=threads) as bx:
            for i, j in T.Parallel(threads, num_per_thread):
                gi = bx * BLK + i * num_per_thread + j
                Y[gi] = X[gi] * 2.0

    return kernel


def tail_kernel(N, threads=256):
    """The remainder `[full, N)` is a second, separate launch."""
    BLK = threads * 8
    full = (N // BLK) * BLK

    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        # One block's worth of threads, one element each, fully predicated.
        with T.Kernel(1, threads=threads) as bx:
            for i in T.Parallel(threads):
                gi = full + i
                if gi < N:
                    Y[gi] = X[gi] * 2.0

    return kernel
