#!/usr/bin/env python3
"""Read a TileLang cache directory and report what is in it, without TileLang.

Chapter 26 claims a specific on-disk layout for the kernel cache and the device
binary cache. This script is the reader for that claim: it walks a cache tree,
re-derives the entry paths, and re-runs the same manifest check that
``KernelCache._verify_cache_dir_integrity`` performs, using only the standard
library.

Two modes
---------
* default: inspect a real cache directory (default ``$TILELANG_CACHE_DIR`` or
  ``~/.tilelang/cache``). If it does not exist, say so and exit 2.
* ``--selftest``: build a throwaway fixture under a temp directory that mimics
  the layout, then assert the reader classifies a complete entry as ok, a
  truncated binary as corrupt, and a missing required file as incomplete.

The fixture is constructed by this script from the layout documented in
``tilelang/cache/kernel_cache.py``. It proves the reader's logic is right; it
does not prove that TileLang writes that layout. The latter is established by
reading the source, not by running anything here.

Usage
-----
    python3 inspect_cache_dir.py
    python3 inspect_cache_dir.py --cache-dir /tmp/tl-cache
    python3 inspect_cache_dir.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# --- layout constants, transcribed from the locked tree ---------------------
# tilelang/cache/kernel_cache.py:51-58
DEVICE_KERNEL_NAME = "device_kernel.cu"
HOST_KERNEL_NAME = "host_kernel.cu"
PARAMS_NAME = "params.json"
MANIFEST_NAME = "manifest.json"
KERNEL_CACHE_SUBDIR = "kernels"
STAGING_SUBDIR = ".staging"
# per-adapter names: tilelang/jit/adapter/kernel_cache.py:8 (tvm_ffi),
# cython/kernel_cache.py (kernel_lib.so), nvrtc/kernel_cache.py:8,
# cutedsl/kernel_cache.py:14; Ascend suffix: tilelang/ascend/kernel_cache.py:11
LIB_CANDIDATES = ("kernel_lib.so", "executable.so", "kernel.cubin", "kernel.py")
ASCEND_DEVICE_KERNEL_NAME = "device_kernel.asc"
# tilelang/cache/binary_cache.py:61-66, cuda_binary_cache.py:17, ascend_binary_cache.py:17-20
BINARY_CACHE_DIRS = {"cuda-binaries": "cubin", "ascend-binaries": "aibin"}


@dataclass
class EntryReport:
    """What we found for one ``<sha256>/`` cache entry."""

    path: Path
    present: list[str] = field(default_factory=list)
    lib_name: str | None = None
    manifest_ok: bool | None = None
    manifest_reason: str = ""
    complete: bool = False
    missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.complete:
            return f"incomplete (missing: {', '.join(sorted(self.missing))})"
        if self.manifest_ok:
            return f"ok   lib={self.lib_name}  files={len(self.present)}"
        return f"corrupt (manifest: {self.manifest_reason})"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_files(entry: Path, lib_name: str | None) -> list[Path]:
    """``KernelCache._get_required_files``: the library plus params.json."""
    if lib_name is None:
        return [entry / PARAMS_NAME]
    return [entry / lib_name, entry / PARAMS_NAME]


def verify_manifest(entry: Path, lib_name: str | None) -> tuple[bool, str]:
    """Mirror of ``KernelCache._verify_cache_dir_integrity`` (kernel_cache.py:807)."""
    manifest_path = entry / MANIFEST_NAME
    try:
        files = json.loads(manifest_path.read_text())["files"]
        entries = list(files.items())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, f"unreadable manifest ({type(exc).__name__})"

    required_names = {p.name for p in required_files(entry, lib_name)}
    if not required_names.issubset(files):
        return False, f"manifest does not cover required files {sorted(required_names)}"

    for name, meta in entries:
        path = entry / name
        try:
            if path.stat().st_size != meta["size"]:
                return False, f"{name}: size mismatch"
            if name in required_names and sha256_file(path) != meta["sha256"]:
                return False, f"{name}: sha256 mismatch"
        except (OSError, KeyError, TypeError) as exc:
            return False, f"{name}: {type(exc).__name__}"
    return True, ""


def inspect_entry(entry: Path) -> EntryReport:
    report = EntryReport(path=entry)
    files = [p for p in entry.iterdir() if p.is_file()]
    report.present = sorted(p.name for p in files)
    have = {p.name for p in files}
    for name in LIB_CANDIDATES:
        if name in have:
            report.lib_name = name
            break

    if DEVICE_KERNEL_NAME not in have and ASCEND_DEVICE_KERNEL_NAME not in have:
        report.missing.append("device_kernel.<cu|asc>")
    if HOST_KERNEL_NAME not in have:
        report.missing.append(HOST_KERNEL_NAME)
    if MANIFEST_NAME not in have:
        report.missing.append(MANIFEST_NAME)
    if report.lib_name is None:
        report.missing.append("kernel_lib.so|executable.so|kernel.cubin|kernel.py")
    if PARAMS_NAME not in have:
        report.missing.append(PARAMS_NAME)

    report.complete = not report.missing
    if report.complete:
        report.manifest_ok, report.manifest_reason = verify_manifest(entry, report.lib_name)
    return report


def inspect_namespace(ns: Path) -> None:
    kernels = ns / KERNEL_CACHE_SUBDIR
    staging = ns / STAGING_SUBDIR
    print(f"namespace: {ns}")
    print(f"  kernels/ entries : {len(list(kernels.iterdir())) if kernels.is_dir() else 0}")
    print(f"  .staging/ entries: {len(list(staging.iterdir())) if staging.is_dir() else 0}")
    if kernels.is_dir():
        for entry in sorted(p for p in kernels.iterdir() if p.is_dir()):
            print(f"    {entry.name[:16]}...  {inspect_entry(entry).summary()}")
    for name, fmt in BINARY_CACHE_DIRS.items():
        root = ns / name
        if not root.is_dir():
            continue
        entries = sorted(p for p in root.iterdir() if p.is_dir())
        print(f"  {name}/ entries: {len(entries)} (device binary cache)")
        for entry in entries[:5]:
            payload = entry / f"kernel.{fmt}"
            meta_path = entry / "metadata.json"
            verdict = "missing payload/metadata"
            if payload.is_file() and meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    ok = (
                        meta.get("size") == payload.stat().st_size
                        and meta.get("sha256") == sha256_file(payload)
                    )
                    verdict = f"ok ({fmt}, {payload.stat().st_size} bytes)" if ok else "corrupt"
                except (OSError, ValueError):
                    verdict = "metadata unreadable"
            print(f"    {entry.name[:16]}...  {verdict}")


def inspect_cache_dir(root: Path) -> int:
    if not root.is_dir():
        print(f"cache directory not found: {root}", file=sys.stderr)
        print("hint: run a TileLang program once, or pass --cache-dir", file=sys.stderr)
        return 2
    print(f"cache root: {root}")
    namespaces = sorted(p for p in root.iterdir() if p.is_dir())
    if not namespaces:
        print("  (empty: no version/host-machine namespace yet)")
        return 0
    for ns in namespaces:
        # Layout is <version>/<host-platform>-<machine>/; also print one level
        # deeper so both shapes are visible on a real cache.
        inspect_namespace(ns)
        for inner in sorted(p for p in ns.iterdir() if p.is_dir() and p.name not in {KERNEL_CACHE_SUBDIR, STAGING_SUBDIR, *BINARY_CACHE_DIRS}):
            inspect_namespace(inner)
    return 0


def _write_manifest(entry: Path) -> None:
    files = {}
    for p in sorted(entry.iterdir()):
        if p.is_file() and p.name != MANIFEST_NAME:
            files[p.name] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    (entry / MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "files": files}, indent=2, sort_keys=True)
    )


def _make_entry(entry: Path, *, truncate_lib: bool = False, drop_params: bool = False) -> None:
    entry.mkdir(parents=True)
    (entry / DEVICE_KERNEL_NAME).write_text('extern "C" __global__ void k() {}\n')
    (entry / HOST_KERNEL_NAME).write_text("int32_t k() { return 0; }\n")
    if not drop_params:
        (entry / PARAMS_NAME).write_text('{"params": []}\n')
    (entry / "kernel_lib.so").write_bytes(b"\x7fELF" + b"\x00" * 4096)
    _write_manifest(entry)
    if truncate_lib:
        with (entry / "kernel_lib.so").open("r+b") as fh:
            fh.truncate(64)


def selftest() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tl-cache-fixture-"))
    try:
        ns = tmp / "0.1.15" / "darwin-arm64"
        ok = ns / KERNEL_CACHE_SUBDIR / ("a" * 64)
        bad = ns / KERNEL_CACHE_SUBDIR / ("b" * 64)
        partial = ns / KERNEL_CACHE_SUBDIR / ("c" * 64)
        _make_entry(ok)
        _make_entry(bad, truncate_lib=True)
        _make_entry(partial, drop_params=True)
        (ns / STAGING_SUBDIR / "leftover").mkdir(parents=True)

        bin_entry = ns / "ascend-binaries" / ("d" * 64)
        bin_entry.mkdir(parents=True)
        payload = b"aibin-payload" * 16
        (bin_entry / "kernel.aibin").write_bytes(payload)
        (bin_entry / "metadata.json").write_text(
            json.dumps(
                {
                    "format": "tilelang.ascend-binary-cache.v1",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )

        failures = []
        r_ok = inspect_entry(ok)
        if not (r_ok.complete and r_ok.manifest_ok):
            failures.append(f"complete entry misclassified: {r_ok.summary()}")
        r_bad = inspect_entry(bad)
        if not (r_bad.complete and not r_bad.manifest_ok):
            failures.append(f"truncated entry misclassified: {r_bad.summary()}")
        r_partial = inspect_entry(partial)
        if r_partial.complete or PARAMS_NAME not in r_partial.missing:
            failures.append(f"partial entry misclassified: {r_partial.summary()}")

        print("--- fixture tree ---")
        inspect_namespace(ns)
        print("--- selftest ---")
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}")
            return 1
        print("ok: reader classifies complete / corrupt / incomplete entries and reads the binary cache")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: $TILELANG_CACHE_DIR or ~/.tilelang/cache)"
    )
    parser.add_argument("--selftest", action="store_true", help="build a fixture and check the reader's logic")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    root = args.cache_dir or os.environ.get("TILELANG_CACHE_DIR") or str(Path.home() / ".tilelang" / "cache")
    return inspect_cache_dir(Path(root))


if __name__ == "__main__":
    raise SystemExit(main())
