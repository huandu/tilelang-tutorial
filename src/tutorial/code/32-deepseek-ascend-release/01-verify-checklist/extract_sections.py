#!/usr/bin/env python3
"""Extract named sections from one or more component READMEs.

This is the companion tool for chapter 32's "怎么自己核验" section. It does
**not** judge whether a component is usable; it only copies the parts of a
README that the judgement has to be based on, so the reader can see the exact
wording instead of a paraphrase.

Usage:
    python3 extract_sections.py README.md [README2.md ...]
    python3 extract_sections.py --sections News,Ongoing README.md
    python3 extract_sections.py --list README.md

Only the Python standard library is used, so it runs anywhere (no NPU/GPU
needed). Missing sections are reported as "(not found)" on purpose: a README
that never mentions `Ongoing` is different from one that says there is no
ongoing work, and the reader needs to see that difference.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_SECTIONS = ("News", "Requirements", "Ongoing")

# A section runs from its own heading to the next heading of the same or a
# higher level (fewer or equal '#' characters).
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def split_sections(text: str) -> list[tuple[int, str, list[str]]]:
    """Return [(level, title, body_lines)] in file order."""
    sections: list[tuple[int, str, list[str]]] = []
    current: tuple[int, str, list[str]] | None = None
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match is None:
            if current is not None:
                current[2].append(line)
            continue
        if current is not None:
            sections.append(current)
        current = (len(match.group(1)), match.group(2), [])
    if current is not None:
        sections.append(current)
    return sections


def find_section(sections, name: str):
    wanted = name.strip().lower()
    for index, (level, title, body) in enumerate(sections):
        if wanted in title.strip().lower():
            # Trim the body at the next heading of the same or higher level.
            trimmed: list[str] = []
            for line in body:
                match = HEADING_RE.match(line)
                if match is not None and len(match.group(1)) <= level:
                    break
                trimmed.append(line)
            return index, level, title, trimmed
    return None


def render(path: Path, sections, names) -> str:
    out: list[str] = [f"=== {path}"]
    for name in names:
        found = find_section(sections, name)
        if found is None:
            out.append(f"## {name}: (not found)")
            continue
        _, level, title, body = found
        out.append(f"{'#' * level} {title}")
        body = [line for line in body]
        while body and body[0].strip() == "":
            body.pop(0)
        while body and body[-1].strip() == "":
            body.pop()
        out.extend(body)
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readmes", nargs="+", help="README files to inspect")
    parser.add_argument(
        "--sections",
        default=",".join(DEFAULT_SECTIONS),
        help=f"comma-separated section names (default: {','.join(DEFAULT_SECTIONS)})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="only print the headings found in each file",
    )
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.sections.split(",") if name.strip() != ""]
    if not names:
        print("no section names given", file=sys.stderr)
        return 2

    failed = 0
    for name in args.readmes:
        path = Path(name)
        if not path.is_file():
            print(f"=== {path}\n    !! file not found", file=sys.stderr)
            failed += 1
            continue
        sections = split_sections(path.read_text(encoding="utf-8", errors="replace"))
        if args.list:
            print(f"=== {path}")
            for level, title, _ in sections:
                print(f"  {'#' * level} {title}")
            continue
        print(render(path, sections, names))
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
