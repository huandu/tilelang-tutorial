# 示例 2：读取缓存目录

## 目的

把“缓存目录长什么样”从一句描述变成一次可重复的读取：

* 递归列出 `$TILELANG_CACHE_DIR/<version>/<platform>-<machine>/` 下的
  `kernels/` 与 `.staging/`；
* 对每个 `<sha256>/` 条目复刻 `KernelCache._verify_cache_dir_integrity`
  （`_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:807-835`）的判定：
  清单里每个文件的**大小**必须一致，被列为必需的文件（adapter 的二进制 + `params.json`）
  还要**内容哈希**一致；
* 列出 `cuda-binaries/` 与 `ascend-binaries/`，并按
  `tilelang/cache/binary_cache.py:125-147` 的规则校验
  `metadata.json` 里的 `size` / `sha256`。

它不 import tilelang，只按源码里写死的路径约定去读磁盘。

## 运行前提

* Python 3.9 以上（只用标准库）。
* 模式一：一个真实的 TileLang 缓存目录（跑过一次 TileLang 程序即可产生）。
* 模式二（`--selftest`）：什么都不需要，脚本自己造 fixture。
* **不需要** GPU / NPU / tilelang / torch。

## 运行命令

```bash
cd chapters/26-runtime-jit-abi/code/02-cache-dir-inspect

# 读真实缓存（默认 $TILELANG_CACHE_DIR，未设置则 ~/.tilelang/cache）
python3 inspect_cache_dir.py
python3 inspect_cache_dir.py --cache-dir /tmp/tl-cache

# 不需要真实缓存：造一份 fixture，验证读取逻辑
python3 inspect_cache_dir.py --selftest
```

## 期望输出

`--selftest` 会打印 fixture 树与三条判据：

```text
--- fixture tree ---
namespace: /var/folders/.../tl-cache-fixture-xxxx/0.1.15/darwin-arm64
  kernels/ entries : 3
  .staging/ entries: 1
    aaaaaaaaaaaaaaaa...  ok   lib=kernel_lib.so  files=5
    bbbbbbbbbbbbbbbb...  corrupt (manifest: kernel_lib.so: size mismatch)
    cccccccccccccccc...  incomplete (missing: params.json)
  ascend-binaries/ entries: 1 (device binary cache)
    dddddddddddddddd...  ok (aibin, 208 bytes)
--- selftest ---
ok: reader classifies complete / corrupt / incomplete entries and reads the binary cache
```

## 正确性判据

* 退出码 0；
* 三条判据全部命中：完整条目判为 `ok`、被截断的条目判为 `corrupt`、
  缺 `params.json` 的条目判为 `incomplete`。

## 这个示例**没有**证明什么

fixture 是脚本按源码里读到的路径约定自己造的。所以本示例验证的是**读取逻辑**，
不是“TileLang 真的这样落盘”。后者由 `01-runtime-anchor-check` 的源码锚点 +
`tilelang/cache/kernel_cache.py:552-640` 的阅读共同支撑，属于静态核对。

## 作者实际做到的验证级别

**在本机运行过（2026-10-01）**：`--selftest` 输出上述内容、退出码 0；
默认模式在本机返回退出码 2 并打印“cache directory not found: ~/.tilelang/cache”，
这与“本机没有安装 tilelang、从未产生缓存”一致。
