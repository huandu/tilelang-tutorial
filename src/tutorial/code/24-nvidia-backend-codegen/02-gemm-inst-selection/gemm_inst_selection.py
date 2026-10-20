#!/usr/bin/env python3
"""A tiny, dependency-free model of TileLang's CUDA GEMM instruction selection.

Mirrors Gemm::SelectInst in
_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375 and the f16/bf16 branch of
GetTCGEN5MMAMeta in _vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h:22-177.

It is NOT the compiler. See README.md for the exact simplifications.
"""

from __future__ import annotations

from dataclasses import dataclass

MMA = "cuda.mma"
FMA = "cuda.fma"
WGMMA = "cuda.wgmma"
TCGEN05 = "cuda.tcgen05"

F16, BF16, F32, TF32, FP8, INT8 = "f16", "bf16", "f32", "tf32", "fp8", "int8"


def arch_of(target: str) -> int:
    return int(target.removeprefix("sm_").rstrip("af"))


def is_volta(target: str) -> bool:
    return 70 <= arch_of(target) < 75


def is_turing(target: str) -> bool:
    return 75 <= arch_of(target) < 80


def is_hopper(target: str) -> bool:
    return 90 <= arch_of(target) < 100


def is_sm100(target: str) -> bool:
    return 100 <= arch_of(target) <= 110


# --- GetTCGEN5MMAMeta, f16/bf16 -> f32 branch only --------------------------

def tcgen5_meta_f16(m: int, n: int, k: int, *, disable_2cta: bool = False,
                    disable_ws: bool = False) -> tuple | None:
    """Return (atom_m, atom_n, atom_k, enable_ws, enable_2cta) or None."""
    if k % 16 != 0:
        return None
    if not disable_2cta:
        if m % 128 == 0:
            for atom_n in range(256, 15, -16):
                if n % atom_n == 0:
                    return (256, atom_n, 16, False, True)
        elif m % 64 == 0:
            for atom_n in range(256, 15, -16):
                if n % atom_n == 0:
                    return (128, atom_n, 16, False, True)
    if m % 128 == 0:
        for atom_n in range(256, 7, -8):
            if n % atom_n == 0:
                return (128, atom_n, 16, False, False)
        return None
    if m % 64 == 0:
        if not disable_ws:
            for atom_n in (256, 128, 64):
                if n % atom_n == 0:
                    return (64, atom_n, 16, True, False)
        return None
    if m % 32 == 0:
        if not disable_ws:
            for atom_n in (256, 128, 64):
                if n % atom_n == 0:
                    return (32, atom_n, 16, True, False)
        return None
    return None


@dataclass(frozen=True)
class GemmOp:
    a_dtype: str
    b_dtype: str
    c_dtype: str
    a_scope: str  # "shared" | "shared.dyn" | "local.fragment" | "shared.tmem"
    b_scope: str
    c_scope: str
    m: int
    n: int
    k: int
    block_size: int = 256
    trans_a: bool = False
    trans_b: bool = False
    explicit: str | None = None  # None | "wgmma" | "tcgen05"
    use_2cta: bool = False
    disable_wgmma: bool = False
    gemm_blockscaled: bool = False


def is_shared(scope: str) -> bool:
    return scope in ("shared", "shared.dyn")


def is_fragment(scope: str) -> bool:
    return scope == "local.fragment"


def is_tmem(scope: str) -> bool:
    return scope == "shared.tmem"


def check_wgmma(op: GemmOp) -> bool:
    """Mirror CheckWgmma (gemm.cc:40-78)."""
    if not is_shared(op.b_scope):
        return False
    if op.c_dtype == F16:
        if op.a_dtype == F16 and op.b_dtype == F16:
            return op.k % 16 == 0
        if op.a_dtype == FP8 and op.b_dtype == FP8:
            return (not op.trans_a) and op.trans_b and op.k % 32 == 0
        return False
    if op.c_dtype == F32:
        if op.a_dtype == F16 and op.b_dtype == F16:
            return op.k % 16 == 0
        if op.a_dtype == BF16 and op.b_dtype == BF16:
            return op.k % 16 == 0
        if op.a_dtype == TF32 and op.b_dtype == TF32:
            return (not op.trans_a) and op.trans_b and op.k % 8 == 0
        if op.a_dtype == FP8 and op.b_dtype == FP8:
            return (not op.trans_a) and op.trans_b and op.k % 32 == 0
        return False
    if op.c_dtype == "int32":
        ok_dtypes = {(INT8, INT8), (INT8, "uint8"), ("uint8", INT8), ("uint8", "uint8")}
        if (op.a_dtype, op.b_dtype) in ok_dtypes:
            return (not op.trans_a) and op.trans_b and op.k % 32 == 0
        return False
    return False


def allow_wgmma(op: GemmOp, target: str) -> bool:
    if op.disable_wgmma:
        return False
    num_warps = op.block_size // 32
    return (is_hopper(target) and op.m >= 64 and num_warps % 4 == 0
            and check_wgmma(op))


def allow_tcgen5_mma(op: GemmOp, target: str) -> bool:
    """Mirror AllowTcgen5Mma (gemm.cc:80-87)."""
    scope_ok = ((is_shared(op.a_scope) or is_tmem(op.a_scope))
                and is_shared(op.b_scope) and is_tmem(op.c_scope))
    if not is_sm100(target) or not scope_ok:
        return False
    if op.c_dtype != F32 or op.a_dtype not in (F16, BF16):
        return False  # only the f16/bf16 -> f32 branch is modeled
    meta = tcgen5_meta_f16(op.m, op.n, op.k)
    return meta is not None


def allow_volta_mma(op: GemmOp) -> bool:
    """Mirror AllowVoltaMma (gemm.cc:99-117)."""
    scope_ok = (is_shared(op.a_scope) or is_fragment(op.a_scope)) and is_shared(op.b_scope)
    if not scope_ok or op.trans_a:
        return False
    if op.a_dtype != F16 or op.b_dtype != F16:
        return False
    if op.c_dtype not in (F16, F32):
        return False
    return op.m % 16 == 0 and op.n % 16 == 0 and op.k % 4 == 0


def allow_turing_mma(op: GemmOp) -> bool:
    """Mirror AllowTuringMma (gemm.cc:131-145)."""
    if op.a_dtype != op.b_dtype:
        return False
    if op.a_dtype == F16:
        return op.c_dtype in (F16, F32)
    if op.a_dtype in (INT8, "uint8", "int4", "uint4"):
        return op.c_dtype == "int32"
    return False


class GemmSelectionError(Exception):
    pass


def select_inst(op: GemmOp, target: str) -> str:
    """Mirror Gemm::SelectInst (gemm.cc:336-375)."""
    if op.explicit == "wgmma":
        if not allow_wgmma(op, target):
            raise GemmSelectionError("T.wgmma_gemm() requires Hopper WGMMA lowering")
        return WGMMA
    if op.explicit == "tcgen05":
        if not allow_tcgen5_mma(op, target):
            raise GemmSelectionError("T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering")
        return TCGEN05
    if op.use_2cta:
        if not allow_tcgen5_mma(op, target):
            raise GemmSelectionError(
                "use_2cta=True requires Blackwell TCGEN5MMA lowering; "
                "no one-CTA instruction fallback is valid")
        return TCGEN05
    if allow_tcgen5_mma(op, target):
        return TCGEN05
    if allow_wgmma(op, target):
        return WGMMA
    if is_volta(target) and not allow_volta_mma(op):
        return FMA
    if is_turing(target) and not allow_turing_mma(op):
        return FMA
    return MMA


def describe(op: GemmOp, target: str) -> str:
    try:
        return select_inst(op, target)
    except GemmSelectionError as exc:
        return "ERROR: " + str(exc)


def self_check(rows):
    expected = {
        "hopper big": WGMMA,
        "hopper small M": MMA,
        "hopper wrong warp group": MMA,
        "hopper fp8 mn-major A": MMA,
        "hopper fp8 k-major A": WGMMA,
        "blackwell C in tmem": TCGEN05,
        "blackwell C in fragment": MMA,
        "blackwell M=64 C in tmem": TCGEN05,
        "turing f16": MMA,
        "volta f16->f16": MMA,
        "volta transposed A -> fma": FMA,
        "ampere": MMA,
        "explicit wgmma on hopper": WGMMA,
        "explicit wgmma on ampere": "ERROR",
        "use_2cta on blackwell": TCGEN05,
        "use_2cta on hopper": "ERROR",
    }
    ok = True
    for name, _op, _target, result in rows:
        want = expected[name]
        good = result.startswith("ERROR") if want == "ERROR" else result == want
        if not good:
            print(f"SELF-CHECK FAILED: {name}: expected {want}, got {result}")
            ok = False
    print("self-check:", "ok" if ok else "FAILED")
    return ok


def main() -> int:
    base = dict(a_dtype=F16, b_dtype=F16, c_dtype=F32, a_scope="shared.dyn",
                b_scope="shared.dyn", c_scope="local.fragment",
                m=128, n=128, k=64)
    scenarios = [
        ("hopper big", GemmOp(**{**base, "trans_b": True}), "sm_90a"),
        ("hopper small M", GemmOp(**{**base, "m": 32, "trans_b": True}), "sm_90a"),
        ("hopper wrong warp group", GemmOp(**{**base, "block_size": 160, "trans_b": True}), "sm_90a"),
        ("hopper fp8 mn-major A",
         GemmOp(a_dtype=FP8, b_dtype=FP8, c_dtype=F32, a_scope="shared.dyn",
                b_scope="shared.dyn", c_scope="local.fragment",
                m=128, n=128, k=64, trans_a=True, trans_b=True), "sm_90a"),
        ("hopper fp8 k-major A",
         GemmOp(a_dtype=FP8, b_dtype=FP8, c_dtype=F32, a_scope="shared.dyn",
                b_scope="shared.dyn", c_scope="local.fragment",
                m=128, n=128, k=64, trans_a=False, trans_b=True), "sm_90a"),
        ("blackwell C in tmem", GemmOp(**{**base, "c_scope": "shared.tmem"}), "sm_100a"),
        ("blackwell C in fragment", GemmOp(**base), "sm_100a"),
        ("blackwell M=64 C in tmem", GemmOp(**{**base, "m": 64, "c_scope": "shared.tmem"}), "sm_100a"),
        ("turing f16", GemmOp(**base), "sm_75"),
        ("volta f16->f16", GemmOp(**{**base, "c_dtype": F16}), "sm_70"),
        ("volta transposed A -> fma", GemmOp(**{**base, "trans_a": True}), "sm_70"),
        ("ampere", GemmOp(**base), "sm_80"),
        ("explicit wgmma on hopper", GemmOp(**{**base, "explicit": "wgmma", "trans_b": True}), "sm_90a"),
        ("explicit wgmma on ampere", GemmOp(**{**base, "explicit": "wgmma"}), "sm_80"),
        ("use_2cta on blackwell",
         GemmOp(**{**base, "c_scope": "shared.tmem", "use_2cta": True}), "sm_100a"),
        ("use_2cta on hopper", GemmOp(**{**base, "use_2cta": True}), "sm_90a"),
    ]
    print("=== T.gemm 指令选择（模型） ===")
    rows = []
    for name, op, target in scenarios:
        result = describe(op, target)
        rows.append((name, op, target, result))
        print(f"{target:<8} M={op.m:<4} N={op.n:<4} K={op.k:<4} "
              f"{op.a_dtype}->{op.c_dtype:<5} A={op.a_scope:<14} C={op.c_scope:<14} "
              f"-> {result}")
    print()
    return 0 if self_check(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
