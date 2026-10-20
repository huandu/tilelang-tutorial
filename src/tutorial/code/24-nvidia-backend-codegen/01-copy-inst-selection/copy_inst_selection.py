#!/usr/bin/env python3
"""A tiny, dependency-free model of TileLang's CUDA copy-instruction selection.

This mirrors the decision order of SelectCopyInstForLowering in
_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-891.

It is NOT the compiler. It is a readable restatement of the documented rules so
that you can poke at the inputs and see how the choice changes. See README.md
for the exact simplifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Copy instruction names, matching CopyInstToString in copy_analysis.cc.
NORMAL = "Normal"
LDSM = "LDSM"
STSM = "STSM"
BULK_LOAD = "BulkLoad"
BULK_STORE = "BulkStore"
BULK_LOAD_1D = "BulkLoad1D"
BULK_STORE_1D = "BulkStore1D"
CP_ASYNC = "CPAsync"
TMEM_LOAD = "TMemLoad"
TMEM_STORE = "TMemStore"


class SelectionError(Exception):
    """An explicit hint could not be honored. The compiler would LOG(FATAL)."""


@dataclass(frozen=True)
class CopyOp:
    """The subset of a T.copy tile op that the selection logic looks at."""

    src_scope: str  # global | shared | shared.dyn | local.fragment | shared.tmem
    dst_scope: str
    dtype_bits: int
    last_dim_extent: int
    src_strides: tuple[int, ...] = ()  # element strides, innermost last
    dst_strides: tuple[int, ...] = ()
    has_layout_map: bool = True
    # Whether the global side of the region is fully contiguous, which the 1D
    # (descriptor-less) TMA path needs. A 2D tile slice of a bigger matrix is
    # NOT contiguous.
    global_region_contiguous: bool = True
    # annotations
    is_tma_copy: bool = False
    is_async_copy: bool = False
    force_cp_async: bool = False
    pipeline_managed: bool = False  # InjectSoftwarePipeline set the marker
    prefer_instruction: str | None = None  # "tma" | "cp_async" | "sync"
    disable_tma: bool = False
    cluster_mask: int = 0
    extra: dict = field(default_factory=dict)


def parse_arch(arch: str) -> int:
    """Turn 'sm_90a' / 'sm_100f' into 90 / 100."""
    return int(arch.removeprefix("sm_").rstrip("af"))


def target_has_bulk_copy(arch: str) -> bool:
    return parse_arch(arch) >= 90


def target_has_async_copy(arch: str) -> bool:
    return parse_arch(arch) >= 80


def target_has_ldmatrix(arch: str) -> bool:
    return parse_arch(arch) >= 75


def target_has_stmatrix(arch: str, is_m16n8: bool = False) -> bool:
    value = parse_arch(arch)
    return value >= 100 if is_m16n8 else value >= 90


def target_has_tmem(arch: str) -> bool:
    return 100 <= parse_arch(arch) <= 110


def is_global(scope: str) -> bool:
    return scope == "global"


def is_shared(scope: str) -> bool:
    return scope in ("shared", "shared.dyn")


def is_fragment(scope: str) -> bool:
    return scope == "local.fragment"


def is_tmem(scope: str) -> bool:
    return scope == "shared.tmem"


def check_global_strides(strides: tuple[int, ...], dtype_bits: int) -> bool:
    """Mirror CheckGlobalStrides (copy_analysis.cc:143-193) for concrete sizes."""
    if not strides:
        return True
    if strides[-1] != 1:
        return False
    for stride in strides[:-1]:
        stride_bytes = stride * dtype_bits // 8
        if stride_bytes % 16 != 0:
            return False
        if stride_bytes >= (1 << 40):
            return False
    return True


def check_bulk_load(op: CopyOp, arch: str, check_last_dim: bool) -> bool:
    if not target_has_bulk_copy(arch):
        return False
    if not is_global(op.src_scope) or not is_shared(op.dst_scope):
        return False
    if check_last_dim and (op.last_dim_extent * op.dtype_bits) % 128 != 0:
        return False
    return check_global_strides(op.src_strides, op.dtype_bits)


def check_bulk_store(op: CopyOp, arch: str, check_last_dim: bool) -> bool:
    if not target_has_bulk_copy(arch):
        return False
    if not is_shared(op.src_scope) or not is_global(op.dst_scope):
        return False
    if check_last_dim and (op.last_dim_extent * op.dtype_bits) % 128 != 0:
        return False
    return check_global_strides(op.dst_strides, op.dtype_bits)


def _flipped(op: CopyOp) -> CopyOp:
    return replace(op, src_scope=op.dst_scope, dst_scope=op.src_scope,
                   src_strides=op.dst_strides, dst_strides=op.src_strides)


def check_bulk_copy_1d(op: CopyOp, arch: str) -> bool:
    """A coarse stand-in for CheckBulkCopy1D (copy_analysis.cc:326-365)."""
    if not check_bulk_load(op, arch, check_last_dim=False):
        return False
    if not op.has_layout_map:
        return False
    if not op.global_region_contiguous:
        return False
    total_bits = op.last_dim_extent * op.dtype_bits
    return total_bits % 128 == 0


def select_tma_inst(op: CopyOp, arch: str, *, allow_load: bool,
                    allow_store: bool, check_last_dim: bool) -> str | None:
    """Mirror SelectTmaInst (copy_analysis.cc:593-610)."""
    if allow_load and check_bulk_copy_1d(op, arch):
        return BULK_LOAD_1D
    if allow_store and check_bulk_copy_1d(_flipped(op), arch):
        return BULK_STORE_1D
    if allow_load and check_bulk_load(op, arch, check_last_dim):
        return BULK_LOAD
    if allow_store and check_bulk_store(op, arch, check_last_dim):
        return BULK_STORE
    return None


def select_sync_like(op: CopyOp, arch: str) -> str:
    """Mirror SelectSyncLikeInst (copy_analysis.cc:612-626)."""
    if target_has_ldmatrix(arch) and is_shared(op.src_scope) and is_fragment(op.dst_scope):
        return LDSM
    if target_has_stmatrix(arch) and is_fragment(op.src_scope) and is_shared(op.dst_scope):
        return STSM
    if target_has_tmem(arch) and is_tmem(op.src_scope) and is_fragment(op.dst_scope):
        return TMEM_LOAD
    if target_has_tmem(arch) and is_fragment(op.src_scope) and is_tmem(op.dst_scope):
        return TMEM_STORE
    return NORMAL


def select_copy_inst(op: CopyOp, arch: str, *, enable_async_copy: bool = True,
                     pass_context_disables_tma: bool = False) -> str:
    """Mirror SelectCopyInstForLowering (copy_analysis.cc:716-807)."""
    if op.cluster_mask != 0:
        inst = select_tma_inst(op, arch, allow_load=True, allow_store=False,
                               check_last_dim=True)
        if inst is None:
            raise SelectionError(
                "cluster_mask requires descriptor-based TMA (kBulkLoad), "
                "but the copy does not meet TMA bulk-load constraints")
        return inst

    if op.is_tma_copy:
        inst = select_tma_inst(op, arch, allow_load=True, allow_store=True,
                               check_last_dim=True)
        if inst is None:
            raise SelectionError("T.tma_copy() cannot be lowered to TMA")
        return inst

    if op.is_async_copy or op.force_cp_async or op.pipeline_managed:
        if not target_has_async_copy(arch):
            raise SelectionError("cp.async requires sm_80+")
        if not is_global(op.src_scope) or not is_shared(op.dst_scope):
            raise SelectionError("cp.async only supports global -> shared")
        return CP_ASYNC

    if op.prefer_instruction == "tma":
        if op.disable_tma:
            raise SelectionError('prefer_instruction="tma" conflicts with disable_tma=True')
        if pass_context_disables_tma:
            raise SelectionError('prefer_instruction="tma" conflicts with tl.disable_tma_lower=True')
        inst = select_tma_inst(op, arch, allow_load=True, allow_store=True,
                               check_last_dim=True)
        if inst is None:
            raise SelectionError('prefer_instruction="tma" could not be honored')
        return inst

    if op.prefer_instruction == "cp_async":
        if not enable_async_copy:
            raise SelectionError('prefer_instruction="cp_async" conflicts with tl.enable_async_copy=False')
        if not (target_has_async_copy(arch) and is_global(op.src_scope) and is_shared(op.dst_scope)):
            raise SelectionError('prefer_instruction="cp_async" could not be honored')
        return CP_ASYNC

    if op.prefer_instruction == "sync":
        return select_sync_like(op, arch)

    # Unannotated: TMA is offered only for the store direction (allow_load=False).
    if not op.disable_tma and not pass_context_disables_tma:
        inst = select_tma_inst(op, arch, allow_load=False, allow_store=True,
                               check_last_dim=True)
        if inst is not None:
            return inst
    return select_sync_like(op, arch)


def _hint(op: CopyOp) -> str:
    if op.cluster_mask:
        return "cluster"
    if op.is_tma_copy:
        return "tma_copy"
    if op.is_async_copy or op.force_cp_async:
        return "async"
    if op.pipeline_managed:
        return "pipeline"
    return op.prefer_instruction or "default"


def run(scenarios):
    rows = []
    for name, op, arch, kwargs in scenarios:
        try:
            result = select_copy_inst(op, arch, **kwargs)
        except SelectionError as exc:
            result = "ERROR: " + str(exc)
        rows.append((name, op, arch, result))
        print(f"{arch:<8} {op.src_scope:>14} -> {op.dst_scope:<14} "
              f"bits={op.dtype_bits:<3} last={op.last_dim_extent:<4} "
              f"hint={_hint(op):<10} -> {result}")
    return rows


def self_check(rows):
    expected = {
        "s->g 2D tile": BULK_STORE,
        "g->s 2D tile, plain T.copy": NORMAL,
        "g->s 2D tile, inside T.Pipelined": CP_ASYNC,
        "g->s 2D tile, prefer=tma": BULK_LOAD,
        "g->s 2D tile, prefer=tma + disable_tma": "ERROR",
        "g->s 2D tile, sm_80": NORMAL,
        "g->s 2D tile, last_dim=30, prefer=tma": "ERROR",
        "g->s 1D contiguous, plain T.copy": NORMAL,
        "g->s 1D contiguous, prefer=tma": BULK_LOAD_1D,
        "s->fragment f16": LDSM,
        "fragment->s f16": STSM,
        "tmem->fragment f32 sm_100": TMEM_LOAD,
        "g->s 2D tile, cluster_mask": BULK_LOAD,
        "g->s 2D tile, cluster_mask + bad align": "ERROR",
    }
    ok = True
    for name, _op, _arch, result in rows:
        want = expected[name]
        good = result.startswith("ERROR") if want == "ERROR" else result == want
        if not good:
            print(f"SELF-CHECK FAILED: {name}: expected {want}, got {result}")
            ok = False
    print("self-check:", "ok" if ok else "FAILED")
    return ok


def main() -> int:
    # A 2D tile slice of a bigger matrix: the global region is NOT contiguous,
    # so the descriptor-based TMA path applies, never the 1D one.
    global_tile = dict(src_strides=(1024, 1), global_region_contiguous=False)
    shared_tile = dict(dst_strides=(1024, 1), global_region_contiguous=False)
    scenarios = [
        ("s->g 2D tile",
         CopyOp("shared.dyn", "global", 16, 64, **shared_tile), "sm_90", {}),
        ("g->s 2D tile, plain T.copy",
         CopyOp("global", "shared.dyn", 16, 64, **global_tile), "sm_90", {}),
        ("g->s 2D tile, inside T.Pipelined",
         CopyOp("global", "shared.dyn", 16, 64, pipeline_managed=True,
                **global_tile), "sm_90", {}),
        ("g->s 2D tile, prefer=tma",
         CopyOp("global", "shared.dyn", 16, 64, prefer_instruction="tma",
                **global_tile), "sm_90", {}),
        ("g->s 2D tile, prefer=tma + disable_tma",
         CopyOp("global", "shared.dyn", 16, 64, prefer_instruction="tma",
                disable_tma=True, **global_tile), "sm_90", {}),
        ("g->s 2D tile, sm_80",
         CopyOp("global", "shared.dyn", 16, 64, **global_tile), "sm_80", {}),
        ("g->s 2D tile, last_dim=30, prefer=tma",
         CopyOp("global", "shared.dyn", 16, 30, prefer_instruction="tma",
                **global_tile), "sm_90", {}),
        ("g->s 1D contiguous, plain T.copy",
         CopyOp("global", "shared.dyn", 16, 64, src_strides=(1,),
                global_region_contiguous=True), "sm_90", {}),
        ("g->s 1D contiguous, prefer=tma",
         CopyOp("global", "shared.dyn", 16, 64, src_strides=(1,),
                global_region_contiguous=True, prefer_instruction="tma"), "sm_90", {}),
        ("s->fragment f16",
         CopyOp("shared.dyn", "local.fragment", 16, 64), "sm_90", {}),
        ("fragment->s f16",
         CopyOp("local.fragment", "shared.dyn", 16, 64), "sm_90", {}),
        ("tmem->fragment f32 sm_100",
         CopyOp("shared.tmem", "local.fragment", 32, 64), "sm_100", {}),
        ("g->s 2D tile, cluster_mask",
         CopyOp("global", "shared.dyn", 16, 64, cluster_mask=0b0011,
                **global_tile), "sm_90", {}),
        ("g->s 2D tile, cluster_mask + bad align",
         CopyOp("global", "shared.dyn", 16, 30, cluster_mask=0b0011,
                **global_tile), "sm_90", {}),
    ]
    print("=== T.copy 搬运指令选择（模型） ===")
    rows = run(scenarios)
    print()
    return 0 if self_check(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
