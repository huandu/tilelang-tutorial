#!/usr/bin/env python3
"""Print the exact procedure for telling a v0.1.15 capability from a main/PR one.

The script is deliberately boring: it prints the commands to run, the order to
run them in, and the question each one answers.  With ``--run`` it also
inspects the local ``_vendor/`` checkouts with read-only git commands, which is
usually enough to discover that they are shallow clones and that comparing two
tags locally therefore needs a real clone (or the corpus main snapshot).

Usage:
    python3 upstream_watch.py                 # print the procedure
    python3 upstream_watch.py --run           # + inspect local checkouts
    python3 upstream_watch.py --repo tile-ai/tilelang

No network access is attempted.  ``--run`` only executes ``git describe``,
``git rev-parse``, ``git rev-list --count`` and ``git status``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TAG = "v0.1.15"
PREVIOUS_TAG = "v0.1.14"

PROCEDURE = """\
判定一个能力属于哪一档，按顺序做这四步，不要跳步：

  第 1 步 · 在锁定 release 里找「符号」，不是找「说法」
      rg -n '<symbol>' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
    命中 -> 「v0.1.15 已具备」，到此结束。
    没命中 -> 继续。

  第 2 步 · 与上一个 release 对照，确认它是不是本次新增
      rg -n '<symbol>' _vendor/tilelang-v0.1.14/tilelang _vendor/tilelang-v0.1.14/src
    两边都命中 -> 说明它在 v0.1.14 之前就落地了，写章节时不要说成「v0.1.15 新增」。

  第 3 步 · 在 main 快照里找（corpus 只读副本）
      rg -n '<symbol>' ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/tilelang \\
                          ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/src
    命中而 release 不命中 -> 「仅 main」：下一个 release 会带上，现在不要依赖。

  第 4 步 · 都没有 -> 只可能是「仅未合入 PR / RFC」
      PR/RFC 快照只能证明「抓取那一刻」的状态，不能证明今天的状态。
      PR 编号 > 快照日期之后合入的，要重新核对；判断不了就写「无法确认」。

四步都做完，再写结论。结论必须带一条可复现命令。
"""

GIT_COMMANDS = """\
要跟踪上游（需要一个完整 clone，不是 --depth 1 的快照）：

  # 0. 准备一个可比较的仓库
  git clone https://github.com/tile-ai/tilelang.git tilelang-upstream
  cd tilelang-upstream && git fetch --tags origin

  # 1. 现在都有哪些 tag，最新的排最前
  git tag --sort=-creatordate | head -20

  # 2. v0.1.15 之后 main 上做了什么（只看昇腾）
  git log --oneline {tag}..origin/main -- tilelang/ascend src/ascend testing/ascend examples/ascend

  # 3. 文件级差异：哪些文件是 tag 之后才出现的
  git diff --stat {tag}..origin/main
  git diff --name-status {tag}..origin/main -- tilelang/ascend src/ascend

  # 4. 某个符号是什么时候进来的
  git log -S '<symbol>' --oneline --all -- tilelang src

  # 5. 未合入的线索（需要 gh CLI）
  gh pr list   --repo tile-ai/tilelang --state open --search 'ascend in:title'
  gh pr list   --repo tile-ai/tilelang --state all  --search 'Ascend 950'
  gh issue list --repo tile-ai/tilelang --state open --label roadmap
  gh issue list --repo tile-ai/tilelang --state all  --search 'RFC in:title'

  # 6. 昇腾适配器仓库（A2/A3，另一棵树）
  gh release list --repo tile-ai/tilelang-ascend
  gh pr list --repo tile-ai/tilelang-ascend --state open

要订阅的东西：
  * Release / tag：tile-ai/tilelang、tile-ai/tilelang-ascend
  * 标签：`Ascend`、`roadmap`、`RFC`、`backend`、`wrong-code`
  * 昇腾官方：CANN 版本说明、bisheng 工具链的 arch 列表（dav-3510 / dav-c310 …）
  * 生态：TileOPs、TileRT、TileFoundry 的 README「News」段落
"""


def run_git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def describe_checkouts(vendor: Path) -> int:
    if not vendor.is_dir():
        print(f"_vendor 不存在：{vendor}", file=sys.stderr)
        return 2
    rows = []
    for child in sorted(vendor.iterdir()):
        if not child.is_dir():
            continue
        head = run_git(child, "rev-parse", "HEAD")
        if head is None:
            rows.append((child.name, "(not a git checkout)", "-", "-"))
            continue
        tag = run_git(child, "describe", "--tags", "--always") or "-"
        count = run_git(child, "rev-list", "--count", "HEAD") or "?"
        status = run_git(child, "status", "--porcelain") or ""
        rows.append((child.name, head[:12], tag, f"{count} commit(s){' dirty' if status else ''}"))
    print(f"{'checkout':<34} | {'HEAD':<12} | {'describe':<28} | history")
    print("-" * 100)
    for name, head, tag, history in rows:
        print(f"{name:<34} | {head:<12} | {tag:<28} | {history}")
    print()
    if all(row[3].startswith("1 commit") for row in rows):
        print(
            "提示：这些是 --depth 1 快照，只有一个 commit，因此本地无法\n"
            "`git diff v0.1.14..v0.1.15`。要比较两个 tag，要么用完整 clone，\n"
            "要么用本目录 02-tree-diff/tree_diff.py 做文件级比较。"
        )
    return 0


def main() -> int:
    here = Path(__file__).resolve()
    book_root = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path, default=book_root)
    parser.add_argument("--repo", default="tile-ai/tilelang", help="upstream repo slug")
    parser.add_argument("--run", action="store_true", help="inspect local _vendor checkouts")
    args = parser.parse_args()

    print(f"# 上游跟踪手册（{args.repo}，本章锁定 {TAG}，对照 {PREVIOUS_TAG}）\n")
    print(PROCEDURE)
    print()
    print(GIT_COMMANDS.format(tag=TAG))

    if args.run:
        print("\n# 本地 _vendor 检查结果\n")
        return describe_checkouts(args.book_root / "_vendor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
