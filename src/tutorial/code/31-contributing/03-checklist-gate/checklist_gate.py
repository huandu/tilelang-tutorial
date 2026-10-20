#!/usr/bin/env python3
"""Fail a self-check list when an unchecked item has no stated reason.

The Ascend PR template asks contributors to explain every skipped item. This
script turns that request into a gate that can run before the PR is opened.

Usage:
    python3 checklist_gate.py <checklist.md>
    python3 checklist_gate.py --template
    python3 checklist_gate.py --self-test
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path


TEMPLATE = """# 改动自检清单

> 未勾选的条目必须写原因或跟踪 issue。

- [ ] 已跑 `bash format.sh`
  - 原因: <填写；不适用时写不适用并给理由>
- [ ] 已跑改动路径对应的测试套件
  - 原因: <填写>
- [ ] 已附性能报告（若改动触及热点路径）
  - 原因: <填写>
- [ ] 已核对后端可移植性（是否需要 requires_cuda / requires_rocm / requires_metal）
  - 原因: <填写>
- [ ] 昇腾改动已按 PR 模板说明双后端情况
  - 原因: <填写>
"""

UNCHECKED = re.compile(r"^\s*-\s*\[\s\]\s*(?P<body>.*)$")
CHECKED = re.compile(r"^\s*-\s*\[[xX]\]\s*")
REASON_INLINE = re.compile(r"(?:—|--)\s*(原因|理由|Because|Reason)\s*[:：]", re.IGNORECASE)
REASON_LINE = re.compile(r"^\s+(?:[-*]\s*)?(原因|理由|Because|Reason)\s*[:：]", re.IGNORECASE)


def audit(text: str) -> list[tuple[int, str]]:
    """Return (line_number, text) for unchecked items with no reason."""
    lines = text.splitlines()
    unexplained: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = UNCHECKED.match(line)
        if match is None:
            continue
        body = match.group("body").strip()
        if REASON_INLINE.search(body):
            continue
        if index + 1 < len(lines) and REASON_LINE.match(lines[index + 1]):
            continue
        unexplained.append((index + 1, body))
    return unexplained


def self_test() -> int:
    failures: list[str] = []

    good = """
- [x] done
- [ ] skipped — 原因: no NPU on this machine
- [ ] other
  - 原因: tracked in #1234
"""
    if audit(good):
        failures.append("well-explained list should pass")

    bad = """
- [ ] skipped silently
"""
    if not audit(bad):
        failures.append("silently skipped item should fail")

    if audit(TEMPLATE):
        failures.append("the bundled template must itself pass")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.md"
        path.write_text(good, encoding="utf-8")
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
        sys.stdout.write(TEMPLATE)
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
    unexplained = audit(path.read_text(encoding="utf-8"))
    if unexplained:
        for line_no, body in unexplained:
            print(f"unexplained: {line_no}: {body}")
        return 1
    print(f"ok: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
