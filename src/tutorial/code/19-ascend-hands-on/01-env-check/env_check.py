#!/usr/bin/env python3
"""第 19 章 §3.5：昇腾环境自检。

这个脚本**只依赖标准库**，因此在没有 NPU、没有装 torch/tilelang 的机器上也能运行；
它逐项探测环境并打印结论，方便你在装机过程中反复确认「卡在哪一层」。

它检查的是「装没装、能不能 import、版本是多少」，**不检查硬件是否真的能算对**——
那需要 §4/§5 的 kernel 示例。

验证级别：CPU 逻辑验证（纯标准库，可在任意机器上复算）。
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _which_binary(name: str) -> Check:
    path = shutil.which(name)
    if path is None:
        return Check(f"binary:{name}", False, "not found on PATH")
    return Check(f"binary:{name}", True, path)


def _env_var(name: str) -> Check:
    value = os.environ.get(name)
    if not value:
        return Check(f"env:{name}", False, "<unset>")
    return Check(f"env:{name}", True, value)


def _import_module(module: str) -> Check:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - 探测脚本要吞掉所有导入异常
        return Check(f"import:{module}", False, f"{type(exc).__name__}: {exc}")
    version = getattr(imported, "__version__", None)
    location = getattr(imported, "__file__", None)
    detail = f"version={version} path={location}"
    return Check(f"import:{module}", True, detail)


def _torch_npu_available() -> Check:
    """只有 import torch_npu 成功之后，torch.npu 才存在。"""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return Check("torch.npu.is_available", False, f"{type(exc).__name__}: {exc}")
    try:
        available = bool(torch.npu.is_available())
    except Exception as exc:  # noqa: BLE001
        return Check("torch.npu.is_available", False, f"{type(exc).__name__}: {exc}")
    detail = "available" if available else "torch.npu exists but reports unavailable"
    return Check("torch.npu.is_available", available, detail)


def collect() -> list[Check]:
    checks: list[Check] = [
        Check("os", True, f"{platform.system()} {platform.machine()} python={sys.version.split()[0]}"),
    ]
    # CANN / 驱动层
    checks.append(_which_binary("npu-smi"))
    checks.append(_which_binary("bisheng"))
    checks.append(_env_var("ASCEND_HOME_PATH"))
    checks.append(_env_var("ASCEND_TOOLKIT_HOME"))
    checks.append(_env_var("TL_ROOT"))
    checks.append(_env_var("ACL_OP_INIT_MODE"))
    # Python 层
    checks.append(_import_module("torch"))
    checks.append(_torch_npu_available())
    checks.append(_import_module("tilelang"))
    # 从这里能看出你拿到的是哪棵树（A 树在 tilelang.language 下，M 树在 tilelang.ascend.language 下）
    checks.append(_import_module("tilelang.language"))
    checks.append(_import_module("tilelang.ascend.language"))
    return checks


def classify(checks: list[Check]) -> str:
    by_name = {c.name: c for c in checks}
    if by_name["import:tilelang"].ok and by_name["torch.npu.is_available"].ok:
        return "看着像一台可用的昇腾开发机；接着按 §3.5 跑 §4/§5 的 kernel 做端到端确认。"
    if by_name["import:tilelang"].ok:
        return "TileLang 装上了，但没有可用的 NPU（或 torch_npu 没配好）：本章示例只能做静态核对/仿真。"
    return "这台机器还没有昇腾环境：先按 §3.2–§3.4 铺驱动/CANN/torch_npu/TileLang。"


def main() -> int:
    checks = collect()
    width = max(len(c.name) for c in checks)
    for check in checks:
        marker = "ok  " if check.ok else "MISS"
        print(f"[{marker}] {check.name:<{width}}  {check.detail}")
    print()
    print("结论：", classify(checks))
    # 这个脚本总是返回 0：它的作用是「描述现状」，不是「判定环境达标」。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
