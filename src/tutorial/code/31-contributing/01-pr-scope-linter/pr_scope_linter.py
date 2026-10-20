#!/usr/bin/env python3
"""Classify a proposed TileLang change and print the gates it must pass.

Offline, stdlib-only. Every gate name below was read out of the locked
checkout `_vendor/tilelang-v0.1.15/`; the file that defines the gate is
recorded next to the gate so the claim can be re-checked.

Usage:
    python3 pr_scope_linter.py <changed-path>...
    git diff --name-only origin/main...HEAD | python3 pr_scope_linter.py -
    python3 pr_scope_linter.py --self-test
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    name: str
    defined_in: str
    command: str


EXAMPLE_GATE = Gate(
    name="examples",
    defined_in=".github/workflows/ci.yml (job `tests`, step `Run examples`)",
    command="cd testing && pytest ../examples",
)
TRANSFORM_GATE = Gate(
    name="transform unit tests",
    defined_in="testing/python/transform/ + .github/workflows/ci.yml (step `Run CUDA tests`)",
    command="python -m pytest testing/python/transform -x",
)
BACKEND_GATE = Gate(
    name="backend contract tests",
    defined_in="testing/python/backend/ + .agents/skills/tilelang-backend/",
    command="python -m pytest testing/python/backend -x",
)
LAYOUT_GATE = Gate(
    name="layout golden harness",
    defined_in="maint/layout_inference/README.md",
    command="python maint/layout_inference/run.py",
)
PERF_GATE = Gate(
    name="performance regression",
    defined_in=".github/workflows/pr-regression-test-bot.yml (comment `@regression-perf`)",
    command="bash maint/scripts/run_perf_regression.sh",
)
ASCEND_GATE = Gate(
    name="ascend examples + unit tests",
    defined_in=".github/workflows/ci.yml (job `ascend`) + maint/scripts/run_local_ci_test_ascend.sh",
    command="bash maint/scripts/run_local_ci_test_ascend.sh",
)
ASCEND_PR_TEMPLATE = Gate(
    name="ascend PR template (dual backend + perf report)",
    defined_in="_vendor/tilelang-ascend-v0.1.2.000/.github/pull_request_template.md",
    command="(manual) fill in the checklist attached to the PR",
)
LINT_GATE = Gate(
    name="format + lint",
    defined_in=".pre-commit-config.yaml + CONTRIBUTING.md",
    command="bash format.sh && pre-commit run --all-files",
)
DOCS_GATE = Gate(
    name="docs build (after merge)",
    defined_in=".github/workflows/publish-docs.yml",
    command="bash maint/scripts/build_docs.sh",
)


def classify(path: str) -> tuple[str, list[Gate]]:
    """Return (category, gates) for one changed path."""
    p = path.strip()
    if not p:
        return "empty", []
    lower = p.lower()
    gates: list[Gate] = []

    if p.startswith("examples/"):
        return "operator / example", [EXAMPLE_GATE, LINT_GATE]

    if p.startswith("src/transform/") or p.startswith("tilelang/transform/"):
        gates = [TRANSFORM_GATE, LINT_GATE]
        if "layout" in lower:
            gates.append(LAYOUT_GATE)
        return "compiler pass", gates

    if p.startswith("src/layout/") or p.startswith("tilelang/layout/"):
        return "layout", [LAYOUT_GATE, TRANSFORM_GATE, LINT_GATE]

    if p.startswith("src/tl_templates/") or "/codegen" in p or p.endswith("codegen.py"):
        return "codegen", [TRANSFORM_GATE, LINT_GATE]

    if p.startswith("tilelang/backend/") or p.startswith("testing/python/backend/"):
        return "backend framework", [BACKEND_GATE, LINT_GATE]

    if re.match(r"^tilelang/[a-z0-9_]+/pipeline\.py$", p):
        return "backend pipeline", [TRANSFORM_GATE, BACKEND_GATE, LINT_GATE]

    if p.startswith("tilelang/") and "/language/" in p:
        return "python frontend / dialect", [TRANSFORM_GATE, LINT_GATE]

    if p.startswith("testing/"):
        return "tests", [LINT_GATE]

    if p.startswith("maint/"):
        gates = [LINT_GATE]
        if "perf" in lower:
            gates.append(PERF_GATE)
        return "maintenance tooling", gates

    if p.startswith(".github/") or p in {"CMakeLists.txt", "pyproject.toml", "setup.py"}:
        return "ci / build", [LINT_GATE]

    if p.startswith("docs/"):
        return "docs", [DOCS_GATE, LINT_GATE]

    return "other", [LINT_GATE]


def annotate_ascend(category: str, gates: list[Gate], path: str) -> tuple[str, list[Gate]]:
    """Ascend paths carry extra obligations on top of their category."""
    if "ascend" not in path.lower():
        return category, gates
    merged = list(gates)
    for extra in (ASCEND_GATE, ASCEND_PR_TEMPLATE):
        if extra not in merged:
            merged.append(extra)
    return f"{category} (Ascend)", merged


def render(paths: list[str]) -> list[str]:
    lines: list[str] = []
    all_gates: list[Gate] = []
    seen: set[str] = set()
    for path in paths:
        category, gates = classify(path)
        category, gates = annotate_ascend(category, gates, path)
        lines.append(f"{path}\n    category: {category}")
        for gate in gates:
            lines.append(f"      - {gate.name}  [{gate.defined_in}]")
            if gate.name not in seen:
                seen.add(gate.name)
                all_gates.append(gate)
        lines.append("")
    lines.append("=" * 72)
    lines.append("Merged gate set (run all of these before pushing):")
    for gate in all_gates:
        lines.append(f"  * {gate.name}")
        lines.append(f"      defined in: {gate.defined_in}")
        lines.append(f"      command:    {gate.command}")
    return lines


def self_test() -> int:
    cases = {
        "examples/softmax/example_softmax.py": "operator / example",
        "src/transform/lower_tile_op.cc": "compiler pass",
        "tilelang/ascend/pipeline.py": "backend pipeline (Ascend)",
    }
    failures: list[str] = []
    for path, expected in cases.items():
        category, gates = annotate_ascend(*classify(path), path)
        if category != expected:
            failures.append(f"{path}: expected {expected!r}, got {category!r}")
        if not gates:
            failures.append(f"{path}: no gates returned")

    ascend_category, ascend_gates = annotate_ascend(
        *classify("tilelang/ascend/pipeline.py"), "tilelang/ascend/pipeline.py"
    )
    names = {gate.name for gate in ascend_gates}
    for required in ("ascend examples + unit tests", "ascend PR template (dual backend + perf report)"):
        if required not in names:
            failures.append(f"Ascend path is missing gate {required!r}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("self-test passed")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if argv == ["--self-test"]:
        return self_test()

    if argv == ["-"]:
        paths = [line for line in sys.stdin.read().splitlines()]
    else:
        paths = argv
    paths = [stripped for stripped in (p.strip() for p in paths) if stripped]
    if not paths:
        print("no paths given", file=sys.stderr)
        return 2

    print("\n".join(render(paths)))
    return 0
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
