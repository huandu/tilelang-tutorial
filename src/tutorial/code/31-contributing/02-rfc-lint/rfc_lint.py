#!/usr/bin/env python3
"""Validate that a TileLang RFC draft contains what reviewers actually ask for.

The locked checkout has no RFC issue template (only bug / feature / question /
release-plan under .github/ISSUE_TEMPLATE/), so RFCs arrive as plain issues.
This linter encodes the sections that the RFCs in
`../tilelang-corpus/08-frontier-unmerged/raw/` all ended up needing.

Usage:
    python3 rfc_lint.py <file.md>
    python3 rfc_lint.py --template
    python3 rfc_lint.py --self-test
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path


REQUIRED = {
    "Motivation": r"motivation|动机|背景",
    "Current Behavior": r"current behavior|现状|当前行为",
    "Proposal": r"proposal|proposed|方案|提案|solution",
    "Compatibility": r"compatib|兼容|影响|blast radius",
    "Testing": r"testing|test plan|验证|测试计划",
    "Risks and Rollback": r"risk|rollback|回退|风险",
    "Non-goals": r"non-goal|not propose|非目标|不做",
}

RECOMMENDED = {
    "Alternatives": r"alternativ|备选|替代方案",
    "Open Questions": r"open question|未决|待定",
    "Staging": r"staging|split|拆分|分阶段|land in small",
}

SKELETON = """# [RFC] <one-line title>

Source: <issue or PR link>
Status: draft | under discussion | accepted | rejected
Touched files (planned): tilelang/language/copy_op.py, testing/python/language/

## Motivation

What problem exists today, and who is blocked by it. Link the issue that
reproduces it.

## Current Behavior

What the compiler/backend does today, with file:line evidence from a specific
commit. If you are guessing, say you are guessing.

## Proposal

The public surface (API, attribute, pass name), the semantics, and the
lowering/codegen implication. Include at least one code block.

## Compatibility and Impact

Blast radius: which backends, which existing kernels, which FFI-visible names.
Call out anything that needs a migration.

## Testing

Which test directories change, which hardware is required, and how a reviewer
can reproduce the result.

## Risks and Rollback

What can break silently, and how the change is reverted.

## Non-goals

What this RFC deliberately does not do.

## Alternatives

Other designs considered and why they lose.

## Open Questions

What is still undecided and who must decide it.

## Staging

How to land this in reviewable pieces.
"""


def lint(text: str) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    weak: list[str] = []
    lowered = text.lower()
    for label, pattern in REQUIRED.items():
        if not re.search(pattern, lowered):
            missing.append(label)
    for label, pattern in RECOMMENDED.items():
        if not re.search(pattern, lowered):
            weak.append(label)

    has_code_block = "```" in text
    mentions_dsl_api = re.search(r"`T\.[A-Za-z_]+", text) or re.search(r"\bT\.[a-z_]+\(", text)
    if mentions_dsl_api and not has_code_block:
        weak.append("Proposal mentions a T.* API but has no fenced code block")

    has_pointer = bool(
        re.search(r"https?://", text)
        or re.search(r"[\w./-]+\.(cc|h|py|md|toml|yml|yaml)\b", text)
        or re.search(r"#\d+", text)
    )
    if not has_pointer:
        weak.append("no file path, link, or issue/PR number anywhere")

    return missing, weak


def self_test() -> int:
    failures: list[str] = []

    missing, weak = lint(SKELETON)
    if missing:
        failures.append(f"template itself is missing sections: {missing}")
    if weak:
        failures.append(f"template itself is weak: {weak}")

    blank = "# [RFC] nothing\n\nJust a sentence with no evidence.\n"
    missing_blank, tough = lint(blank)
    if len(missing_blank) < len(REQUIRED):
        failures.append("empty draft should fail every required section")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rfc.md"
        path.write_text(SKELETON, encoding="utf-8")
        if not path.is_file():
            failures.append("tempfile round-trip failed")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("self-test passed")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--template"]:
        sys.stdout.write(SKELETON)
        return 0
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2

    path = Path(argv[0])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    missing, weak = lint(text)
    if missing:
        for label in missing:
            print(f"missing: {label}")
    if weak:
        for label in weak:
            print(f"weak:    {label}")
    if missing:
        return 1
    print(f"ok: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
