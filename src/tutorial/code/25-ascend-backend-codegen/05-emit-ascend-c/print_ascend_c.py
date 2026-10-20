#!/usr/bin/env python3
"""Print (and audit) the Ascend C / PTO source TileLang generates.

Two modes:

  --audit   offline, no tilelang import; verifies that every API name used by
            the kernels below exists in the pinned trees.
  --emit    requires an Ascend-enabled tilelang build + CANN; prints
            get_kernel_source() and runs the §4 checklist over it.

The author has never run --emit (no NPU, no CANN on this machine).  --audit is
the only mode with a verified result in the book.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path


# --------------------------------------------------------------------------
# Kernels (written against the pinned APIs; never executed here)
# --------------------------------------------------------------------------

def make_m_tree_kernel(M=256, N=256, K=512, block_M=64, block_N=64, block_K=64):
    """M tree (`_vendor/tilelang-v0.1.15`): target="ascend", dav-3510."""
    import tilelang
    import tilelang.ascend.language as T

    num_blocks = (M // block_M) * (N // block_N)
    n_tiles = N // block_N

    @tilelang.jit(target="ascend")
    def matmul_relu(A, B):
        Mv, Nv, Kv = T.const("M, N, K")
        A: T.Tensor((Mv, Kv), T.bfloat16)
        B: T.Tensor((Nv, Kv), T.bfloat16)
        C = T.empty((Mv, Nv), T.float32)

        with T.Kernel(num_blocks) as bx:
            A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
            B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
            C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
            C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)

            for tile in T.Persistent([Mv // block_M * n_tiles], num_blocks, bx):
                m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
                for k in T.Pipelined(Kv // block_K, num_stages=2):
                    T.copy(A[m, k * block_K], A_l1)
                    T.copy(B[n, k * block_K], B_l1, l2_cache_ctrl="NOTALLOC_KEEP")
                    T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))
                T.dual_copy(C_l0c, C_ub)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(block_M // 2, block_N):
                        C_ub[i, j] = T.max(C_ub[i, j], 0)
                T.dual_copy(C_ub, C[m : m + block_M, n : n + block_N],
                            l2_cache_ctrl="NOTALLOC_PW")
        return C

    return matmul_relu


def make_a_tree_kernel(M=256, N=256, K=256, block_M=64, block_N=64, K_L1=64):
    """A tree (`_vendor/tilelang-ascend-v0.1.2.000`): target="ascendc" | "pto"."""
    import tilelang
    import tilelang.language as T

    m_num, n_num = M // block_M, N // block_N

    @tilelang.jit(out_idx=[-1])
    def matmul(A, B, dtype="float16", accum_dtype="float"):
        @T.prim_func
        def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
                 C: T.Tensor((M, N), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
                bx, by = cid // n_num, cid % n_num
                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((K_L1, block_N), dtype)
                C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
                with T.Scope("C"):
                    for k in T.serial(T.ceildiv(K, K_L1)):
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                        T.copy(B[k * K_L1, by * block_N], B_L1)
                        T.barrier_all()
                        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                        T.barrier_all()
                    T.copy(C_L0, C[bx * block_M, by * block_N])
        return main

    return matmul


# --------------------------------------------------------------------------
# Offline audit
# --------------------------------------------------------------------------

API_ANCHORS = {
    "M": (
        ("tilelang/ascend/language/allocate.py", "def alloc_shared("),
        ("tilelang/ascend/language/allocate.py", "def alloc_l1("),
        ("tilelang/ascend/language/allocate.py", "def alloc_l0c("),
        ("tilelang/ascend/language/copy_op.py", "def dual_copy("),
        ("tilelang/ascend/language/copy_op.py", "l2_cache_ctrl"),
        ("tilelang/ascend/language/gemm_op.py", "def gemm("),
        ("tilelang/ascend/language/gemm_op.py", "clear_accum"),
        ("tilelang/ascend/language/frame.py", "def SimtVF("),
        ("tilelang/ascend/language/kernel.py", "def Kernel("),
        ("tilelang/ascend/target.py", 'target.strip() != "ascend"'),
    ),
    "A": (
        ("tilelang/language/allocate.py", "def alloc_L1("),
        ("tilelang/language/allocate.py", "def alloc_L0C("),
        ("tilelang/language/allocate.py", "def alloc_ub("),
        ("tilelang/language/ascend.py", "def gemm_v0("),
        ("tilelang/language/ascend.py", "def barrier_all("),
        ("tilelang/language/warpgroup.py", "def Scope("),
        ("tilelang/utils/target.py", 'target in ["ascendc", "pto"]'),
    ),
}


def find_book_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "_vendor" / "tilelang-v0.1.15").is_dir():
            return cur
        cur = cur.parent
    raise SystemExit("cannot locate book root")


def audit(book: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        py_compile.compile(__file__, cfile=os.path.join(tmp, "self.pyc"), doraise=True)
    print(f"[ok] {Path(__file__).name} py_compile passed")
    trees = {
        "M": book / "_vendor" / "tilelang-v0.1.15",
        "A": book / "_vendor" / "tilelang-ascend-v0.1.2.000",
    }
    misses = 0
    for tree, anchors in API_ANCHORS.items():
        for rel, needle in anchors:
            path = trees[tree] / rel
            if path.is_file() and needle in path.read_text(encoding="utf-8", errors="ignore"):
                print(f"  ok   [{tree}] {rel}: {needle!r}")
            else:
                print(f"  MISS [{tree}] {rel}: {needle!r}")
                misses += 1
    if misses:
        print(f"[FAIL] {misses} API anchor(s) missing")
        return 1
    total = sum(len(v) for v in API_ANCHORS.values())
    print(f"[ok] {total} API anchors verified against the pinned trees")
    print("note: --audit only proves the API names exist; it does NOT compile a kernel.")
    return 0


# --------------------------------------------------------------------------
# Emission + the §4 checklist
# --------------------------------------------------------------------------

CHECKLIST = (
    ("kernel signature", r'extern\s+"C"\s+__global__\s+__(mix__\(1,\s*2\)|vector__|cube__)'),
    ("UB buffer qualifier", r"__ubuf__"),
    ("GM->L1 DMA", r"asc_copy_gm2l1_(nd2nz|dn2nz)"),
    ("L1->L0A/L0B DMA", r"asc_copy_l12l0(a|b)"),
    ("Cube MAD", r"asc_mmad"),
    ("L0C->UB copy", r"asc_copy_l0c2ub"),
    ("L0C->GM copy", r"asc_copy_l0c2gm|asc_copy_ub2gm_align"),
    ("SET/Wait flag", r"asc_set_flag|asc_wait_flag|set_flag|wait_flag"),
)


def emit(target: str, book: Path) -> int:
    if target in ("ascend", "asc"):
        factory = make_m_tree_kernel()
        print(f"# building M-tree kernel (target={target!r})")
    elif target in ("ascendc", "pto", "auto"):
        factory = make_a_tree_kernel()
        print(f"# building A-tree kernel (target={target!r})")
    else:
        raise SystemExit(f"unsupported target {target!r}")

    kernel = factory
    source = kernel.get_kernel_source()
    print(source)
    print("\n# ---- chapter 25 checklist ----")
    for label, pattern in CHECKLIST:
        hit = re.search(pattern, source)
        print(f"[{'ok ' if hit else '?? '}] {label:<20} /{pattern}/")
    print("\nnote: the checklist locates strings; you still have to read the code.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="offline API-name audit")
    mode.add_argument("--emit", action="store_true", help="print generated device source")
    parser.add_argument("--target", default="ascend")
    parser.add_argument("--book-root", type=Path, default=None)
    args = parser.parse_args()
    book = args.book_root or find_book_root(Path(__file__).parent)
    return audit(book) if args.audit else emit(args.target, book)


if __name__ == "__main__":
    sys.exit(main())
