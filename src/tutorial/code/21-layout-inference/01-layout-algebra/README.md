# 01 · layout 代数（纯 Python）

## 目的

不依赖 TileLang，用纯 Python 复算第 21 章 §2 的四个概念：

1. **双射**：`[8,8]` 转置映射 `(i,j) -> (j,i)` 是否一一对应；
2. **非双射 + padding**：`(i,j) -> i*10 + j` 的物理下标空间与空洞；
3. **往返（round-trip）**：为什么丢位的映射会让 `Layout::inverse()` 抛
   `NormalizeIterException`（对应 `_vendor/tilelang-v0.1.15/src/layout/layout.cc:1150-1160`）；
4. **fragment**：复算 `MakeGemmFragment8x8`
   （`_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:45-52`）的
   `forward_thread = j//2 + 4*i`、`index = j%2`，验证“32 个线程、每线程 2 个槽位”
   且无碰撞。

## 运行前提

* Python 3.9+，**只用标准库**（`itertools`、`collections`）。
* 不需要 GPU / NPU，不需要 TileLang。

## 运行命令

```bash
python3 layout_algebra.py
```

## 预期输出（本机实跑，Python 3.11）

```text
transpose [8,8]: bijective=True, points=64
padded [8,8] -> stride 10: used=64, holes=16, is_surjective=False, collisions=0
broken map: collisions=48, round-trip mismatch=48
MakeGemmFragment8x8: threads=32, slots_per_thread={2}, bijective=True
```

（上面的碰撞数与 mismatch 计数是脚本的稳定输出。）

## 验证级别

**CPU 逻辑验证**：本机（macOS / Apple Silicon，Python 3.11）实跑通过。
输出是**排列模型**的结果，不是任何真实硬件上的耗时或 bank conflict 次数。
