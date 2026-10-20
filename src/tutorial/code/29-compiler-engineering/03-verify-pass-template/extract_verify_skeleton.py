#!/usr/bin/env python3
"""把锁定版本里两个 verify pass 的「骨架」抽出来，带真实行号。

不 import tilelang，不需要 GPU/NPU，只用标准库解析两个 C++ 文件：

    _vendor/tilelang-v0.1.15/src/transform/verify_buffer_init.cc
    _vendor/tilelang-v0.1.15/src/transform/verify_parallel_loop.cc

对每个文件打印：

  * 文件头注释的起止行；
  * 所有 ``VisitStmt_`` / ``VisitExpr_`` 重载（行号 + 签名）；
  * ``EmitReport`` 的定义行；
  * ``Pass X()`` 定义行与 ``CreatePrimFuncPass`` 调用行；
  * ``TVM_FFI_STATIC_INIT_BLOCK`` 块与其中的注册名。

用途：写自己的 verify pass 时，先把这份骨架打印出来对着填，比通读一遍
几百行实现更快定位「哪几块是必须抄的」。

验证级别：**CPU 逻辑验证**（本机 macOS、无 tilelang，2026-10-01 运行通过）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[4]
VENDOR = BOOK_ROOT / "_vendor" / "tilelang-v0.1.15"
TARGETS = {
    "verify_buffer_init": VENDOR / "src" / "transform" / "verify_buffer_init.cc",
    "verify_parallel_loop": VENDOR / "src" / "transform" / "verify_parallel_loop.cc",
}

# 注意用 [ \t]* 而不是 \s*：\s 会跨行匹配，导致 ^ 锚点落在前一行的换行上，
# 行号就会凭空少 1——这类 off-by-one 正是写 IR 分析工具时最烦的 bug。
VISIT_RE = re.compile(r"^[ \t]*(void|Stmt|PrimExpr|bool)[ \t]+Visit(Stmt|Expr)_\(", re.M)
EMIT_RE = re.compile(r"^[ \t]*void[ \t]+EmitReport[ \t]*\(", re.M)
PASS_RE = re.compile(r"^[ \t]*(?:tvm::transform::Pass|Pass)[ \t]+(\w+)[ \t]*\(", re.M)
CREATE_RE = re.compile(r"CreatePrimFuncPass\([^;]*\)", re.M)
FFI_RE = re.compile(r"TVM_FFI_STATIC_INIT_BLOCK\(\)\s*\{(?P<body>.*?)\n\}", re.S)
REGISTER_RE = re.compile(r'refl::GlobalDef\(\)\.def\(\s*"([^"]+)"')


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def header_span(lines: list[str]) -> tuple[int, int]:
    """返回文件顶部 ``/*! ... */`` 注释的行号区间；没有就返回 (0, 0)。"""
    if not lines or not lines[0].lstrip().startswith("/*!"):
        return (0, 0)
    for index, line in enumerate(lines):
        if line.strip().endswith("*/"):
            return (1, index + 1)
    return (1, len(lines))


def describe(path: Path) -> list[str]:
    if not path.is_file():
        return [f"  [MISSING] {path}"]

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []

    start, end = header_span(lines)
    if start == 0:
        out.append("  header comment       : (none)")
    else:
        out.append(f"  header comment       : lines {start}-{end} ({end - start + 1} lines)")
    out.append(f"  total lines          : {len(lines)}")

    out.append("  visitor overrides:")
    found_visits = False
    for match in VISIT_RE.finditer(text):
        found_visits = True
        snippet = lines[line_of(text, match.start()) - 1].strip()
        out.append(f"    L{line_of(text, match.start()):>4}  {snippet}")
    if not found_visits:
        out.append("    (none)")

    emit = EMIT_RE.search(text)
    out.append(f"  EmitReport           : L{line_of(text, emit.start()) if emit else '-'}")

    for match in PASS_RE.finditer(text):
        name = match.group(1)
        if name in {"VerifyBufferInit", "VerifyParallelLoop"}:
            out.append(f"  pass factory         : L{line_of(text, match.start())}  Pass {name}()")

    for match in CREATE_RE.finditer(text):
        snippet = " ".join(match.group(0).split())
        out.append(f"  CreatePrimFuncPass   : L{line_of(text, match.start())}  {snippet}")

    ffi = FFI_RE.search(text)
    if ffi:
        out.append(f"  FFI block            : lines {line_of(text, ffi.start())}-"
                   f"{line_of(text, ffi.end())}")
        for registered in REGISTER_RE.finditer(ffi.group("body")):
            out.append(f"    registered name    : {registered.group(1)}")
    else:
        out.append("  FFI block            : (none)")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(TARGETS), help="只打印其中一个")
    args = parser.parse_args()

    names = [args.only] if args.only else sorted(TARGETS)
    for name in names:
        print(f"=== {name} ===")
        print(f"    source: {TARGETS[name].relative_to(BOOK_ROOT)}")
        for line in describe(TARGETS[name]):
            print(line)
        print()

    print("对照要点：两个 pass 的骨架完全相同，只有三处不同——")
    print("  1) 遍历时关心的节点（BufferStore/BufferLoad 还是 For(parallel)+BufferStore）；")
    print("  2) 判定「有问题」的规则（无写入就报，还是并行写同一地址就报）；")
    print("  3) EmitReport 给出的补救建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
