#!/usr/bin/env python3
"""第 13 章 · API 名核对（纯标准库，CPU 可跑）。

问题：本章的示例分属两个仓库、两个方言，写错 API 名（尤其是把主干没有的
`T.tile.*` / `T.alloc_ub` 写进「主干」示例里）在无卡机器上不会被发现。

做法：把示例代码里出现的 `T.` 前缀属性访问（含一层子命名空间）全部抓出来，
再跟锁定版本 `_vendor/` 里的名字集合对一遍。

**这是一个启发式检查，不是类型检查。** 名字集合由静态扫描 `tilelang/language/**`
下的 `def` / 赋值 / `__all__` / `import` 语句拼出来，动态注册的名字（比如靠
`getattr` 注入的）可能漏掉，因此：

* 报「未找到」的名字要人工复核，再决定是代码写错还是扫描漏了；
* 报「已找到」不代表签名对得上——参数与 dtype 约束仍以正文引用的源文件为准。

用法
----

    # 主干方言：检查本章所有示例
    python3 api_audit.py --root ../.. \
        --vendor ../../../../../_vendor/tilelang-v0.1.15 \
        --label mainline

    # 适配器方言：只检查昇腾适配器骨架
    python3 api_audit.py --root ../../06-ascend \
        --vendor ../../../../../_vendor/tilelang-ascend-v0.1.2.000 \
        --label adapter --only ascend_adapter_sketch.py
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*from\s+[\w.]+\s+import\s+(.+)$", re.MULTILINE)
ALL_RE = re.compile(r"__all__\s*=\s*[\[(]([^\])]*)", re.DOTALL)

# 不是 tilelang 语言面的名字：T.Tensor 之类的类型标注、以及 tvm 的通用工具。
IGNORED_NAMES = {
    "const",  # eager-only，单独在 ANNOTATIONS 里说明
}


def collect_language_names(vendor: Path) -> tuple[set[str], dict[str, Path]]:
    """扫描 `tilelang/language/**`，返回所有出现过的标识符与它所在的文件。"""

    names: set[str] = set()
    where: dict[str, Path] = {}
    roots = [
        vendor / "tilelang" / "language",
        vendor / "tilelang" / "cuda" / "language",
        vendor / "tilelang" / "ascend" / "language",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            found: set[str] = set()
            found |= set(DEF_RE.findall(text))
            found |= set(ASSIGN_RE.findall(text))
            for block in ALL_RE.findall(text):
                found |= set(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", block))
                found |= set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?=\s*[,)\]])", block))
            for block in IMPORT_RE.findall(text):
                block = block.split("#")[0]
                block = block.replace("(", " ").replace(")", " ")
                for token in block.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    if " as " in token:
                        token = token.split(" as ")[-1].strip()
                    if token and token != "*":
                        found.add(token)
            for name in found:
                names.add(name)
                where.setdefault(name, path)
    return names, where


def scan_usages(path: Path) -> set[tuple[str, str | None]]:
    """用 tokenizer 抓 `T.<name>` / `T.<ns>.<name>`，**跳过注释与字符串**。

    跳过字符串是必要的：正文和 docstring 里会引用 `T.` 名字做说明，那些不是
    真的在调用；如果一起抓进来，检查结果就会充满假阳性。
    """

    hits: set[tuple[str, str | None]] = set()
    with path.open("rb") as fh:
        tokens = list(tokenize.tokenize(fh.readline))
    for i, tok in enumerate(tokens):
        if tok.type != tokenize.NAME or tok.string != "T":
            continue
        if i + 2 >= len(tokens):
            continue
        if tokens[i + 1].string != "." or tokens[i + 2].type != tokenize.NAME:
            continue
        base = tokens[i + 2].string
        sub: str | None = None
        if (
            i + 4 < len(tokens)
            and tokens[i + 3].string == "."
            and tokens[i + 4].type == tokenize.NAME
        ):
            sub = tokens[i + 4].string
        if base not in IGNORED_NAMES:
            hits.add((base, sub))
    return hits


def collect_usages(root: Path, only: list[str] | None) -> dict[Path, set[tuple[str, str | None]]]:
    usages: dict[Path, set[tuple[str, str | None]]] = {}
    for path in sorted(root.rglob("*.py")):
        if only and path.name not in only:
            continue
        try:
            hits = scan_usages(path)
        except (tokenize.TokenError, SyntaxError, UnicodeDecodeError) as exc:  # pragma: no cover
            print(f"  ! 无法分词 {path}: {exc}")
            continue
        if hits:
            usages[path] = hits
    return usages


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=here.parent, help="要检查的代码根目录")
    parser.add_argument("--vendor", type=Path, required=True, help="锁定版本的 tilelang 仓库根目录")
    parser.add_argument("--label", default="mainline", help="打印用的方言名")
    parser.add_argument("--only", nargs="*", help="只检查这些文件名")
    parser.add_argument(
        "--absent",
        nargs="*",
        default=[],
        help="断言这些名字**不在**该方言的语言面里（用来复现「主干没有 T.alloc_ub」这类结论）",
    )
    parser.add_argument(
        "--absent-namespace",
        nargs="*",
        default=[],
        help="断言这些子命名空间没有被 `from . import X as <name>` 导出（例如主干没有 `T.tile`）",
    )
    args = parser.parse_args()

    if not args.vendor.is_dir():
        print(f"skipped: vendor not found at {args.vendor}")
        return 0
    if not args.root.is_dir():
        print(f"skipped: root not found at {args.root}")
        return 0

    names, where = collect_language_names(args.vendor)
    if not names:
        print(f"skipped: no language names collected under {args.vendor}")
        return 0

    usages = collect_usages(args.root, args.only)
    flat = sorted({base for hits in usages.values() for base, _ in hits})
    nested = sorted({f"{base}.{sub}" for hits in usages.values() for base, sub in hits if sub})

    total = sum(len(h) for h in usages.values())
    print(f"# API 核对（方言：{args.label}）")
    print(f"vendor         : {args.vendor}")
    print(f"扫描到的名字集合: {len(names)} 个")
    print(f"示例文件        : {len(usages)} 个，共 {total} 处 T.* 用法")
    print()
    print("## 检查的文件")
    for path in sorted(usages):
        rel = path.relative_to(args.root.parent) if args.root.parent in path.parents else path
        print(f"  {rel}")
    print()

    missing_flat = [n for n in flat if n not in names]
    print("## 一层的 `T.` 属性访问")
    print(f"  用到 {len(flat)} 个：" + ", ".join(flat))
    if missing_flat:
        print(f"  ⚠ 未在 vendor 名字集合里找到 {len(missing_flat)} 个：{', '.join(missing_flat)}")
        print("    这可能是代码写错，也可能是扫描没覆盖到动态注册的名字——需要人工复核。")
    else:
        print("  ✓ 全部命中")
    print()

    failed = bool(missing_flat)
    if args.absent:
        print("## 断言：这些名字不存在")
        for name in args.absent:
            if name in names:
                print(f"  ✗ {name} 存在（断言失败）")
                failed = True
            else:
                print(f"  ✓ {name} 确实不在该方言的语言面里")
        print()

    if args.absent_namespace:
        print("## 断言：这些子命名空间没有被导出")
        inits = [
            args.vendor / "tilelang" / "language" / "__init__.py",
            args.vendor / "tilelang" / "ascend" / "language" / "__init__.py",
            args.vendor / "tilelang" / "cuda" / "language" / "__init__.py",
        ]
        for ns in args.absent_namespace:
            pattern = re.compile(rf"import\s+\S+\s+as\s+{re.escape(ns)}\b")
            hits = []
            for init in inits:
                if init.is_file() and pattern.search(init.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(init)
            if hits:
                print(f"  ✗ {ns} 被导出：{', '.join(str(h) for h in hits)}（断言失败）")
                failed = True
            else:
                print(f"  ✓ {ns} 没有被任何语言面 `__init__.py` 导出")
        print()

    print("## 两层：`T.<ns>.<name>`（子命名空间，如 `T.tile.` 与 `T.simd.`）")
    if not nested:
        print("  （无）")
    for dotted in nested:
        base, sub = dotted.split(".", 1)
        ok = base in names and sub in names
        mark = "✓" if ok else "⚠"
        src = where.get(sub)
        origin = f"  ← {src.name}" if src is not None else ""
        print(f"  {mark} {dotted}{origin}")

    print()
    if failed:
        print("结论：有未命中的名字，需人工复核（见上）。")
        return 1
    print("结论：所有 T.* 名字都在锁定版本的扫描集合里（仍不等于签名/约束正确）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
