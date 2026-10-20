# 示例 01：layout 代数小模型（不需要 GPU）

## 目的

用一个**只用标准库**的 Python 脚本复算本章三张图里的排列结果：

1. 32×32 `float` 共享 tile 按列访问时的 bank 冲突（stride=32 vs padding 到 33）；
2. 32B / 64B / 128B 三种 bank swizzle 的 XOR 置换表，并验证它们是**双射**；
3. `MakeGemmFragment8x8` 的线程/寄存器槽映射，并验证每个线程恰好 2 个槽。

它是对硬件行为的**模型**，不是 benchmark：脚本从不声称任何加速比，只统计「一个 warp
访问了多少个不同 bank」以及「映射是不是双射」。

## 运行前提

* 任意 Python 3.9+，**不需要** GPU、NPU、TileLang、PyTorch、numpy。
* 本机（macOS / Apple Silicon、Python 3.11）实际运行过。

## 运行命令

```bash
python3 layout_algebra.py
```

## 预期输出（本机实跑结果，节选）

```text
1) 32x32 float tile, warp reads one column (thread t -> row t, col 0)
  no padding (row length 32)   stride=32  distinct banks= 1  worst per-bank=32  -> 32-way bank conflict
  padded (row length 33)       stride=33  distinct banks=32  worst per-bank= 1  -> conflict-free

2) XOR swizzle unit permutations (unit = 16B; fp16 -> 8 elements)
  128B: 3-bit XOR, row period 8, 8 units per group
        c ->   0  1  2  3  4  5  6  7
    s=0     0  1  2  3  4  5  6  7
    s=1     1  0  3  2  5  4  7  6
   ...
        bijection=True  column spread=8 unit(s) over an 8-row period (group width=8)

3) MakeGemmFragment8x8: thread = 4*i + j//2, local = j%2
  i=0  T00L0  T00L1  T01L0  T01L1  T02L0  T02L1  T03L0  T03L1
   ...
  cells=64  threads=32  slots per thread=[2]  valid bijection=True

all checks passed
```

最后一行是脚本的自校验：三种 swizzle 都是双射、fragment 映射也是双射时打印
`all checks passed`，否则打印 `CHECK FAILED`。

## 公式来源（逐条在锁定版本里核对过）

| 脚本里的函数 | 对应源码 |
| --- | --- |
| `xor2x2` / `xor4x4` / `xor8x8` | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:376-393` |
| 32B / 64B / 128B 的 2D swizzle | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:442-530` |
| `make_{quarter,half,full}_bank_swizzled_layout` 的对外签名 | `_vendor/tilelang-v0.1.15/tilelang/layout/swizzle.py:94-126` |
| `MakeGemmFragment8x8` | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:47-53` |
| bank 宽度 4B、32 banks、连续字到连续 bank | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:1545-1554` |
| 「读一列 → 所有线程撞同一个 bank；padding 到 33 消除」 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:2622-2641` |

## 验证级别

**CPU 逻辑验证**：`python3 -m py_compile layout_algebra.py` 通过，脚本在本机实跑，
输出如上。**不是硬件实测**——它验证的是排列公式本身自洽，不是任何机器上的真实耗时。

## 已知边界

* 32B / 64B swizzle 的分组只有 2 / 4 个 16B 单元，所以「一列在 8 行内的散布」上限是
  分组宽度（2 / 4），不是 8；脚本据此把校验式写成 `散度数 == 分组宽度`，而不是硬编码 8。
* 脚本只覆盖 bank 冲突的**共享内存**模型；不涉及 L1 命中、合并访存或指令发射。
