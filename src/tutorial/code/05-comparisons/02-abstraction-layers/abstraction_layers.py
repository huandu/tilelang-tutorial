"""Chapter 05's coordinate system, encoded as runnable, dependency-free Python.

This does NOT measure anything. It is a lookup + decision helper for the two
tables in the chapter:
  * the abstraction-level table (chapter section 2.2);
  * the "when to choose what" decision table (chapter section 5).

Every row carries an evidence id that matches the chapter's references.md
(S<n> = locked source tree, R<n> = corpus material).

Verification level: logic verified on CPU (pure Python, no third-party deps).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    level: str          # "thread/instruction" | "tile" | "compiler substrate"
    writes: str
    compiler_does: str
    relation: str       # relation to TileLang
    evidence: str


TOOLS: tuple[Tool, ...] = (
    Tool(
        "CUDA",
        "thread/instruction",
        "grid/block/warp/thread mapping, __shared__, __syncthreads()",
        "lowers SPMD C++ to PTX/SASS; no tile-level inference",
        "complementary (TileLang's NVIDIA backend emits CUDA C)",
        "R7,R8",
    ),
    Tool(
        "Triton",
        "tile",
        "a block-level program (tl.load/tl.dot/tl.store)",
        "tile-level passes: coalescing, smem alloc + sync, pipelining",
        "competitor (same tile layer)",
        "R2,R4",
    ),
    Tool(
        "CUTLASS/CuTe",
        "thread/instruction",
        "C++ Layout/Tensor/atoms; CuTe DSL mirrors CuTe C++",
        "layout algebra and indexing; no automatic tiling policy",
        "complementary (T.gemm lowers onto CuTe/MMA)",
        "R5,R6,S1",
    ),
    Tool(
        "TVM",
        "compiler substrate",
        "(not an operator front end) TE/TIR/TIRx + schedules",
        "IR, pass infra, target/device abstraction, autotuning",
        "substrate (TileLang is built on TVM)",
        "R10,R11,R12,S10,S11",
    ),
    Tool(
        "cuTile",
        "tile",
        "Python tile program (per third-party description)",
        "unknown locally -- INSUFFICIENT MATERIAL",
        "competitor (NVIDIA tile layer)",
        "R17,R9",
    ),
    Tool(
        "Ascend C",
        "thread/instruction",
        "__global__ __aicore__ kernel; TPipe/TQue; CopyIn/Compute/CopyOut",
        "bisheng compiles CCE C++ to AI Core instructions; no tile inference",
        "competitor + complementary (a TileLang lowering target)",
        "R14,S3",
    ),
    Tool(
        "PTO-DSL",
        "thread/instruction",
        "pto.load / tile.extract / tile.matmul / pto.store",
        "PTOAS InsertSync auto-inserts sync; auto software pipelining",
        "competitor + complementary (TileLang can target PTO-ISA)",
        "R17,R18,R19",
    ),
)

LEVELS: tuple[str, ...] = ("tile", "thread/instruction", "compiler substrate")


def print_table() -> None:
    print("=" * 78)
    print("第 05 章 · 抽象层次表")
    print("=" * 78)
    for t in TOOLS:
        print(f"\n[{t.name}]  层次={t.level}  证据={t.evidence}")
        print(f"  用户写什么 : {t.writes}")
        print(f"  编译器替你做: {t.compiler_does}")
        print(f"  与 TileLang  : {t.relation}")


def print_levels() -> None:
    print("\n" + "=" * 78)
    print("三层抽象（从低到高）")
    print("=" * 78)
    for level in ("thread/instruction", "tile", "compiler substrate"):
        members = [t.name for t in TOOLS if t.level == level]
        print(f"  {level:<22}: {', '.join(members)}")
    print("\n  横切说明：'compiler substrate' 不在纵轴上，它是横穿两级的底盘。")


def choose(hardware: str, goal: str, control: str) -> str:
    """Decision helper mirroring chapter section 5.

    hardware: "nvidia" | "ascend" | "either"
    goal:     "operator" | "compiler"
    control:  "low" (want explicit control) | "high" (let the compiler decide)
    """
    hw = hardware.lower()
    gl = goal.lower()
    ct = control.lower()

    if gl == "compiler":
        return (
            "TVM (substrate) + TileLang (to add passes). "
            "Evidence: S10/S11/S12, R10/R11/R12."
        )
    if hw == "ascend":
        if ct == "low":
            return (
                "Ascend C or PTO-DSL (explicit L1/L0/UB). "
                "Evidence: R14, R17, R18/R19."
            )
        return (
            "TileLang@Ascend (automatic layout/schedule/sync). "
            "Evidence: S3 (tilelang/ascend/README.md:141-151), S11."
        )
    if hw == "nvidia":
        if ct == "low":
            return (
                "CUTLASS/CuTe (incl. CuTe DSL) for library-grade control. "
                "Evidence: R5, R6."
            )
        return (
            "Triton (compiler owns the schedule) or TileLang "
            "(you keep the dataflow). Evidence: R2/R4 and S1/S10."
        )
    return (
        "Cross-vendor: TileLang (one frontend, per-backend lowering). "
        "Evidence: S2 (backend support table), S13."
    )


def main() -> None:
    print_table()
    print_levels()

    print("\n" + "=" * 78)
    print("决策示例")
    print("=" * 78)
    demos = (
        ("ascend", "operator", "high"),
        ("nvidia", "operator", "low"),
        ("either", "compiler", "high"),
    )
    for hardware, goal, control in demos:
        ans = choose(hardware, goal, control)
        print(f"  ({hardware}, {goal}, control={control}) -> {ans}")


if __name__ == "__main__":
    main()
