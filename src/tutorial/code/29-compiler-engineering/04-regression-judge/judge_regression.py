#!/usr/bin/env python3
"""按 TileLang v0.1.15 的判定规则，判断一次性能对比是否算「回归」。

输入是两个 JSON 文件，格式与
``_vendor/tilelang-v0.1.15/maint/scripts/run_current_regression.py`` 的输出一致::

    {"git": {"commit": ..., "dirty": ...},
     "results": {"<case>": {"latency": <float>}, ...},
     "failures": {...}}

判定规则来自 ``maint/scripts/test_perf_regression.py:24-32`` 与
``:243-257``：

    * 逐 case 算 ``speedup = old_latency / new_latency``；
    * 单条低于 ``THRESHOLD``（默认 0.90）**不算**回归——那是噪声；
    * 只有**至少 MIN_COUNT 条**（默认 2）低于阈值，才判为回归。

那段注释给出的标定依据是：PR 机器人到 #3176 为止的 33 份报告、1901 个样本里，
真正的回归都伴随 14–23 条同时塌陷，而 7 份「假警报」都恰好只有一条落在 0.044
这样的极端值、其余全是 1.000。因此「至少两条同时掉」正好把两类分开。

本脚本只做判定，不跑 benchmark（本机没有 GPU/NPU，跑不了）。

验证级别：**CPU 逻辑验证**（本机 macOS，2026-10-01 运行 ``--self-test`` 通过）。

用法::

    python3 judge_regression.py --old base.json --new cur.json
    python3 judge_regression.py --old base.json --new cur.json --fail-on-regression
    python3 judge_regression.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLD = 0.90
DEFAULT_MIN_COUNT = 2


def load_results(path: Path) -> dict[str, float]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError(f"{path}: 'results' must be an object")
    parsed: dict[str, float] = {}
    for name, value in results.items():
        if isinstance(value, dict):
            value = value.get("latency")
        latency = float(value)
        if math.isfinite(latency) and latency > 0.0:
            parsed[str(name)] = latency
    return parsed


def compare(
    old: dict[str, float],
    new: dict[str, float],
    threshold: float,
    min_count: int,
) -> tuple[list[tuple[str, float, float, float]], list[tuple[str, float]]]:
    """返回 (逐 case 明细, 判定为回归的 case)。"""
    rows: list[tuple[str, float, float, float]] = []
    for name in sorted(set(old) & set(new)):
        speedup = old[name] / new[name]
        rows.append((name, old[name], new[name], speedup))
    slow = [(name, speedup) for name, _, _, speedup in rows if speedup < threshold]
    regressed = slow if len(slow) >= min_count else []
    return rows, regressed


def render(rows, regressed, threshold: float, min_count: int, *, show_table: bool = True) -> str:
    lines = [
        f"threshold={threshold:g}  min_count={min_count}  cases={len(rows)}",
        "",
    ]
    if show_table:
        lines.append("| case | old | new | speedup |")
        lines.append("| --- | --- | --- | --- |")
        for name, old_latency, new_latency, speedup in rows:
            lines.append(f"| {name} | {old_latency:.6g} | {new_latency:.6g} | {speedup:.4f}x |")
        lines.append("")
    if regressed:
        lines.append(f"**回归：{len(regressed)}/{len(rows)} 条低于 {threshold:g}x**")
        for name, speedup in regressed:
            lines.append(f"  - {name}: {speedup:.4f}x")
    else:
        below = sum(1 for _, _, _, s in rows if s < threshold)
        lines.append(
            f"**不判为回归**：{below} 条低于 {threshold:g}x，"
            f"未达到 {min_count} 条的门槛。"
        )
    return "\n".join(lines)


def _self_test() -> int:
    """用两组构造数据演示判定规则，输出即正文 6.3 节引用的内容。"""
    # 场景 A：40 条里恰好 1 条 0.85x，其余 1.0x —— 单点噪声，不是回归。
    old_a = {f"case_{i:02d}": 1.0 for i in range(40)}
    new_a = dict(old_a)
    new_a["case_07"] = 1.0 / 0.85
    rows_a, reg_a = compare(old_a, new_a, DEFAULT_THRESHOLD, DEFAULT_MIN_COUNT)
    print("场景 A：40 条里 1 条 0.85x")
    print(render(rows_a, reg_a, DEFAULT_THRESHOLD, DEFAULT_MIN_COUNT, show_table=False))
    print()

    # 场景 B：同样 40 条，但有 2 条掉到 0.88x —— 达到门槛，判为回归。
    new_b = dict(old_a)
    new_b["case_07"] = 1.0 / 0.88
    new_b["case_19"] = 1.0 / 0.88
    rows_b, reg_b = compare(old_a, new_b, DEFAULT_THRESHOLD, DEFAULT_MIN_COUNT)
    print("场景 B：40 条里 2 条 0.88x")
    print(render(rows_b, reg_b, DEFAULT_THRESHOLD, DEFAULT_MIN_COUNT, show_table=False))
    print()

    # 场景 C：同一个 case 只在一个版本里出现 —— 不参与比较。
    rows_c, reg_c = compare({"a": 1.0, "b": 1.0}, {"a": 0.5}, DEFAULT_THRESHOLD, DEFAULT_MIN_COUNT)
    print("场景 C：一侧缺 case")
    print(render(rows_c, reg_c, DEFAULT_THRESHOLD, DEFAULT_MIN_COUNT))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, help="基线结果 JSON")
    parser.add_argument("--new", type=Path, help="当前结果 JSON")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.old is None or args.new is None:
        parser.error("give --old and --new, or --self-test")

    rows, regressed = compare(
        load_results(args.old), load_results(args.new), args.threshold, args.min_count
    )
    print(render(rows, regressed, args.threshold, args.min_count))
    if regressed and args.fail_on_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
