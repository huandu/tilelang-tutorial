#!/usr/bin/env python3
"""Check that every `path:line` / `path:line-line` citation in the chapter resolves.

The locked trees are the only evidence allowed in this book, so a citation
that points past EOF (or at a file that moved) is a real defect, not a typo.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# path:line or path:line-line.  The path must contain a slash so we do not
# mistake prose like "第 20 章:3" for a citation.
CITATION = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\-]*/[A-Za-z0-9_.\-]+):(\d+)(?:-(\d+))?")


def find_book_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "_vendor" / "tilelang-v0.1.15").is_dir():
            return cur
        cur = cur.parent
    raise SystemExit("cannot locate book root")


def search_roots(book: Path, chapter: Path) -> list[Path]:
    """Bases a bare `src/...` / `tilelang/...` citation may be relative to.

    The chapter states once, up front, that M-tree paths are relative to
    `_vendor/tilelang-v0.1.15/`, A-tree paths to `_vendor/tilelang-ascend-v0.1.2.000/`
    and NPUIR paths to the corpus snapshot.  The checker therefore tries every
    locked base and reports an ambiguity when a bare path exists in more than one.
    """
    corpus = book.parent / "tilelang-corpus"
    return [
        chapter,
        book,
        book.parent,                                  # ../tilelang-corpus/...
        book / "_vendor" / "tilelang-v0.1.15",
        book / "_vendor" / "tilelang-ascend-v0.1.2.000",
        corpus,
        corpus / "07-tilelang-ascend-adapter" / "raw",
        corpus / "07-tilelang-ascend-adapter" / "raw" / "tilelang-ascend",
        corpus / "07-tilelang-ascend-adapter" / "raw" / "tilelang-mlir-ascend",
    ]


def resolve(book: Path, chapter: Path, rel: str) -> list[Path]:
    if rel.startswith("_vendor/") or rel.startswith("chapters/") or rel.startswith("tools/"):
        cand = book / rel
        return [cand] if cand.is_file() else []
    if rel.startswith("../"):
        cand = (chapter / rel).resolve()
        if cand.is_file():
            return [cand]
        cand = (book / rel).resolve()
        return [cand] if cand.is_file() else []
    hits: list[Path] = []
    for root in search_roots(book, chapter):
        cand = root / rel
        if cand.is_file() and cand not in hits:
            hits.append(cand)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path, default=None)
    parser.add_argument(
        "--files",
        nargs="*",
        default=["chapter.md", "exercises.md", "solutions.md", "references.md"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    chapter_dir = Path(__file__).resolve().parents[2]
    book = args.book_root or find_book_root(Path(__file__).parent)

    total = 0
    unresolved: list[str] = []
    out_of_range: list[str] = []
    seen: set[str] = set()

    files = [f for f in args.files if (chapter_dir / f).is_file()]
    for name in files:
        text = (chapter_dir / name).read_text(encoding="utf-8")
        for m in CITATION.finditer(text):
            rel, start, end = m.group(1), int(m.group(2)), m.group(3)
            end_i = int(end) if end else start
            key = f"{rel}:{start}-{end_i}"
            if key in seen:
                continue
            seen.add(key)
            total += 1
            line_no = text[: m.start()].count("\n") + 1
            hits = resolve(book, chapter_dir, rel)
            if not hits:
                unresolved.append(f"{name}:{line_no}  {rel}")
                continue
            if len(hits) > 1 and args.verbose:
                print(f"  ~   {rel}: matches {len(hits)} trees")
            # A bare path like tilelang/language/__init__.py exists in all three
            # trees; pick the candidate that actually contains the line range.
            line_counts = [sum(1 for _ in h.open(encoding="utf-8", errors="ignore")) for h in hits]
            valid = [h for h, n in zip(hits, line_counts) if 1 <= start <= end_i <= n]
            if valid:
                path = valid[0]
                n_lines = line_counts[hits.index(path)]
            else:
                path = hits[0]
                n_lines = max(line_counts)
            if start < 1 or start > n_lines or end_i > n_lines or end_i < start:
                out_of_range.append(
                    f"{name}:{line_no}  {rel}:{start}-{end_i} (file has {n_lines} lines)"
                )
            elif args.verbose:
                print(f"  ok  {rel}:{start}-{end_i}")

    print(f"scanned {len(files)} file(s), {total} unique citations")
    for item in unresolved:
        print(f"[FAIL] unresolved: {item}")
    for item in out_of_range:
        print(f"[FAIL] out of range: {item}")
    if unresolved or out_of_range:
        return 1
    print(f"[ok] {total}/{total} resolve, 0 out-of-range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
