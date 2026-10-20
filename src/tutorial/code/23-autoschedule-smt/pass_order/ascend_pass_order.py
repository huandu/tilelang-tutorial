#!/usr/bin/env python3
"""静态核对昇腾 AutoSchedule 的 pass 顺序（不编译、不连 NPU）。

做的事只有两件：

1. 从锁定版本源码 `tilelang/ascend/pipeline.py` 里把
   `if auto_schedule_enabled:` 块内的 pass 调用按顺序抽出来；
2. 在 `src/ascend/transform/` 里找到每个 pass 的 FFI 注册处
   （`refl::GlobalDef().def("tl.transform.<Pass>"`），打印 文件:行号。

用法：
    python3 ascend_pass_order.py                 # 用相对默认路径找 _vendor
    python3 ascend_pass_order.py --repo /path/to/tilelang-v0.1.15

退出码 0 = 顺序与预期一致且每个 pass 都注册过；1 = 有差异（会逐条打印）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXPECTED = [
    "NormalizeControlFlowForSchedule",
    "NormalizeConflictHints",
    "MaterializeScheduleUnits",
    "AnnotateMultiBufferEligible",
    "EstimateLatency",
    "AutoSchedule",
    "AssignCore",
    "PrepareMultiBuffer",
    "ResolveCore",
    "InsertSync",
    "MaterializeMultiBuffer",
    "LowerScheduledTIR",
    "RestoreWhileLoops",
]

CALL_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?"
    r"(?:ascend_transform|tilelang\.transform)\.([A-Za-z_][A-Za-z0-9_]*)\("
)


def default_repo() -> Path:
    # chapters/23-autoschedule-smt/code/pass_order/ -> book root -> _vendor
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "_vendor" / "tilelang-v0.1.15"
        if candidate.is_dir():
            return candidate
    return Path("_vendor/tilelang-v0.1.15")


def extract_autoschedule_block(pipeline_py: Path) -> list[str]:
    lines = pipeline_py.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "if auto_schedule_enabled:":
            start = idx
            break
    if start is None:
        raise SystemExit(f"找不到 `if auto_schedule_enabled:`（{pipeline_py}）")
    calls: list[str] = []
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith(" " * 8):
            break
        match = CALL_RE.match(line)
        if match:
            calls.append(match.group(1))
    return calls


def find_registration(src_dir: Path, pass_name: str) -> tuple[str, int] | None:
    needle_a = f'def("tl.transform.{pass_name}"'
    for path in sorted(src_dir.rglob("*.cc")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if needle_a not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle_a in line:
                return str(path), lineno
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None,
                        help="锁定版本源码根目录（默认自动找 _vendor/tilelang-v0.1.15）")
    args = parser.parse_args()
    repo = args.repo or default_repo()
    pipeline_py = repo / "tilelang" / "ascend" / "pipeline.py"
    src_dir = repo / "src" / "ascend" / "transform"
    if not pipeline_py.is_file():
        return fail(f"找不到 {pipeline_py}；用 --repo 指到 v0.1.15 根目录")

    found = extract_autoschedule_block(pipeline_py)
    print(f"源码：{pipeline_py}")
    print(f"`if auto_schedule_enabled:` 块内按顺序读到 {len(found)} 个 pass\n")

    problems: list[str] = []
    for idx, name in enumerate(found, start=1):
        reg = find_registration(src_dir, name) if src_dir.is_dir() else None
        if reg is None:
            problems.append(f"{name}: 在 {src_dir} 里找不到 tl.transform.{name} 注册")
            where = "注册处未找到"
        else:
            rel = Path(reg[0]).relative_to(repo) if Path(reg[0]).is_absolute() else reg[0]
            where = f"{rel}:{reg[1]}"
        print(f"  {idx:2d}. {name:<30s} {where}")

    if found != EXPECTED:
        problems.append(f"顺序/集合与预期不一致\n  读到: {found}\n  预期: {EXPECTED}")

    print()
    if problems:
        print("FAIL")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("ok: 顺序一致，且每个 pass 都在 src/ascend/transform 里有 FFI 注册")
    return 0


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
