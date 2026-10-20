#!/usr/bin/env python3
"""Chapter 08 / §8: flag Python constructs that the kernel DSL does not accept.

This is a **teaching model**, not TileLang's parser. It walks a Python AST with
the standard library and reports every construct that appears in the table of
``_vendor/tilelang-v0.1.15/docs/programming_guides/python_compatibility.md``,
so you can tell *before* you start a compile whether a snippet is in bounds.

It is deliberately conservative: it looks at the whole file, not just the body
of ``with T.Kernel(...)``, because that is what the table describes ("scripts
inside ``with T.Kernel``"). Nested host-side helpers in the same file will
produce false positives; that is a known and visible limitation.

Run:
    python3 py_compat_scan.py                  # scan the built-in demo snippet
    python3 py_compat_scan.py some_kernel.py   # scan your own file
    python3 py_compat_scan.py --self-test      # check the rules against samples
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass

OK = "OK  "
WARN = "WARN"
BAD = "BAD "


@dataclass
class Finding:
    status: str
    line: int
    construct: str
    note: str


# statuses mirror the three glyphs in the upstream table.
BAD_CALLS = {
    "len": "len() is not available in kernels -- use buffer.shape[dim]",
    "type": "type() is not available in kernels",
    "isinstance": "isinstance() is not available in kernels",
    "print": "print() only works for Python (host) expressions -- use T.print()",
}


class Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self._for_is_python_iterable = False

    def _add(self, status: str, node: ast.AST, construct: str, note: str) -> None:
        self.findings.append(Finding(status, getattr(node, "lineno", 0), construct, note))

    # --- loops -------------------------------------------------------------
    def visit_For(self, node: ast.For) -> None:
        src = ast.unparse(node.iter) if hasattr(ast, "unparse") else ""
        if isinstance(node.iter, ast.Call) and src.startswith("range("):
            self._add(OK, node, "for ... in range(...)", "maps to T.serial(...)")
        elif isinstance(node.iter, (ast.List, ast.Tuple)) or src.startswith(("enumerate(", "zip(")):
            self._add(OK, node, "for ... in <python iterable>", "compile-time expansion; must be finite")
        else:
            self._add(WARN, node, f"for {ast.unparse(node.target)} in {src}",
                      "not recognized as range()/list/tuple/enumerate/zip")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._add(OK, node, "while", "condition must be a TIR expression; constant-true is rejected")
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        self._add(OK, node, "break", "supported in serial/unroll/parallel/while loops")

    def visit_Continue(self, node: ast.Continue) -> None:
        self._add(OK, node, "continue", "supported in serial/unroll/parallel/while loops")

    # --- conditionals ------------------------------------------------------
    def visit_If(self, node: ast.If) -> None:
        self._add(OK, node, "if / elif / else", "python bools fold at trace time; TIR conds stay")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._add(OK, node, "x if cond else y", "ternary expression")
        self.generic_visit(node)

    # --- assignment --------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) > 1:
            self._add(BAD, node, "a = b = c", "chained assignment is not supported; split it")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add(OK, node, f"{ast.unparse(node.target)} {type(node.op).__name__.lower()}= ...",
                  "augmented assignment")
        self.generic_visit(node)

    # --- defs and classes --------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # the outermost @T.prim_func/@tilelang.jit factory is a host function;
        # anything nested is a kernel-level def.
        self._add(WARN, node, f"def {node.name}", "kernels allow no nested functions; use @T.macro")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(BAD, node, f"class {node.name}", "classes are not supported in kernels")
        self.generic_visit(node)

    # --- with / assert / calls --------------------------------------------
    def visit_With(self, node: ast.With) -> None:
        names = []
        for item in node.items:
            src = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
            names.append(src)
        allowed = all(n.startswith(("T.Kernel", "T.ClusterKernel", "T.ws", "T.Persistent")) for n in names)
        self._add(OK if allowed else WARN, node, "with " + ", ".join(names),
                  "only T.Kernel / T.ws style contexts are recognized" if allowed
                  else "not a known DSL context manager")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._add(WARN, node, "assert", "use T.device_assert or T.assert inside a kernel")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BAD_CALLS:
            self._add(BAD, node, f"{node.func.id}()", BAD_CALLS[node.func.id])
        self.generic_visit(node)


DEMO_SNIPPET = '''
import tilelang.language as T


def kernel_factory(N, BLK=128):
    @T.prim_func
    def kernel(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, BLK), threads=128) as bx:
            for i in T.Parallel(BLK):
                gi = bx * BLK + i
                if gi < N:
                    Y[gi] = X[gi] * 2.0
            # deliberately out of bounds for the scanner:
            n = len(X)
            a = b = 1
            with open("x") as fh:
                pass

    return kernel
'''


def scan_source(text: str, label: str) -> list[Finding]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [Finding(BAD, exc.lineno or 0, "<syntax error>", str(exc))]
    scanner = Scanner()
    scanner.visit(tree)
    scanner.findings.sort(key=lambda f: (f.line, f.construct))
    print(f"\n=== {label} ===")
    width = max((len(f.construct) for f in scanner.findings), default=10)
    for f in scanner.findings:
        print(f"  {f.status}  :{f.line:<4} {f.construct:<{width}}  {f.note}")
    bad = sum(1 for f in scanner.findings if f.status == BAD)
    warn = sum(1 for f in scanner.findings if f.status == WARN)
    print(f"  {len(scanner.findings)} findings: {bad} BAD, {warn} WARN")
    return scanner.findings


SELF_TESTS = {
    "len() is BAD": ("def k(T):\n  n = len(X)\n", BAD),
    "chained assign is BAD": ("def k(T):\n  a = b = c\n", BAD),
    "class is BAD": ("class C:\n  pass\n", BAD),
    "range loop is OK": ("def k(T):\n  for i in range(4):\n    pass\n", OK),
    "while is OK": ("def k(T):\n  while i < 4:\n    i += 1\n", OK),
}


def self_test() -> int:
    failures = 0
    for label, (src, expected) in SELF_TESTS.items():
        scanner = Scanner()
        scanner.visit(ast.parse(src))
        got = [f.status for f in scanner.findings]
        ok = expected in got
        print(f"  {'pass' if ok else 'FAIL'}  {label:<24} statuses={got}")
        failures += 0 if ok else 1
    print(f"\n{len(SELF_TESTS) - failures}/{len(SELF_TESTS)} self-tests pass")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="python files to scan")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.paths:
        scan_source(DEMO_SNIPPET, "<built-in demo>")
        return 0
    bad_total = 0
    for path in args.paths:
        text = open(path, encoding="utf-8").read()
        findings = scan_source(text, path)
        bad_total += sum(1 for f in findings if f.status == BAD)
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main())
