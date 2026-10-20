"""第七章 · 安装前的环境体检。

在真机上装 tilelang 之前，先跑一遍这个脚本，把「环境要求」对着官方
``docs/get_started/Installation.md`` 里那几条逐项验一遍。它只报告事实，
不做自动安装、不改任何文件。

运行：
    python check_env.py                    # 只体检 CUDA / 通用环境
    python check_env.py --ascend           # 额外体检昇腾环境

验证级别：**在本机运行过（纯 Python，只读环境）**。脚本本身不需要 GPU；
它检测的是「如果本机没有 GPU，会如实说没有」。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys


def report(name: str, state: bool | None, detail: str, required: bool = False) -> None:
    mark = "ok  " if state else ("FAIL" if state is False else "?   ")
    suffix = "  (必需)" if required else ""
    print(f"  [{mark}] {name:<30} {detail}{suffix}")


def check_python() -> None:
    v = sys.version_info
    report("Python >= 3.10", (v.major, v.minor) >= (3, 10), platform.python_version(), required=True)


def check_glibc() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        ver = libc.gnu_get_libc_version().decode()
        major, minor = (int(x) for x in ver.split(".")[:2])
        report("glibc >= 2.28", (major, minor) >= (2, 28), ver, required=True)
    except Exception:
        report("glibc >= 2.28", None, "非 glibc 环境（macOS / musl 等），跳过")


def check_cuda() -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        report("nvcc（host CUDA 工具链）", False, "PATH 里没有 nvcc；可用 pip 提供的 CUDA（>=13）")
        return
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=20).stdout
        line = next((l for l in out.splitlines() if "release" in l), "").strip()
        report("nvcc 可用", True, f"{nvcc}  {line}")
    except Exception as exc:  # pragma: no cover
        report("nvcc 可用", None, f"找到 {nvcc} 但无法执行：{exc}")


def check_torch() -> None:
    try:
        import torch
    except Exception:
        report("import torch", False, "未安装 torch（wheel 安装会把它作为依赖装上）")
        return
    report("import torch", True, torch.__version__)
    try:
        ok = torch.cuda.is_available()
        report("torch.cuda.is_available()", ok, "有 NVIDIA GPU" if ok else "没有可见的 NVIDIA GPU")
    except Exception as exc:
        report("torch.cuda.is_available()", None, str(exc))


def check_tilelang() -> None:
    try:
        import tilelang
    except Exception as exc:
        report("import tilelang", False, f"{type(exc).__name__}: {exc}")
        return
    report("import tilelang", True, getattr(tilelang, "__version__", "?"))
    try:
        from tilelang import env

        report("TILELANG_CACHE_DIR", True, str(env.TILELANG_CACHE_DIR))
    except Exception:
        pass


def check_ascend() -> None:
    print("\n== 昇腾环境 ==")
    home = os.environ.get("ASCEND_HOME_PATH")
    report("ASCEND_HOME_PATH", bool(home), home or "未设置（build_wheel_ascend.sh 需要它）")
    bisheng = shutil.which("bisheng")
    report("bisheng（编译器）", bisheng is not None, bisheng or "PATH / BISHENG_HOME/bin 里没有 bisheng")
    lld = shutil.which("ld.lld")
    report("ld.lld（CCE 版）", lld is not None, lld or "PATH 里没有 ld.lld")
    try:
        import torch
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401

        ok = torch.npu.is_available()
        report("torch.npu.is_available()", ok, "有昇腾 NPU" if ok else "没有可见的昇腾 NPU")
    except Exception as exc:
        report("import torch_npu", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ascend", action="store_true", help="额外体检昇腾环境")
    args = parser.parse_args()

    print(f"平台：{platform.platform()} / {platform.machine()}\n")
    print("== 通用环境（对照 docs/get_started/Installation.md:7-9） ==")
    check_python()
    check_glibc()
    print("\n== CUDA 环境 ==")
    check_cuda()
    print("\n== Python 依赖 ==")
    check_torch()
    check_tilelang()
    if args.ascend:
        check_ascend()
    print("\n提示：本脚本只报告事实。安装命令见本章 3 节，安装失败见本章 8 节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
