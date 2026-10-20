# 02 · 版本数与 arena 打包模型

## 目的

把 `InjectSoftwarePipeline` 与 `MergeSharedMemoryAllocations` 的两步决策写成可复算的
算术：

1. 每个 buffer 的版本数 = `use_stage - def_stage + 1`
   （`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1488-1542`）；
2. 物化后每个 buffer var 的活跃区间，交给一个 linear-scan + best-fit freelist 的
   arena 打包器（`_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc`，
   `FreeList` 与 `LinearScanPack`）。

它回答两个问题：**“这个 num_stages 还放得下吗”** 和
**“哪些 buffer 能复用同一段 shared memory”**。

## 运行前提

Python 3.9+，只用标准库。不需要 GPU/NPU，不需要 TileLang。

## 运行命令

```bash
python3 buffer_versions.py                        # 默认 scenario=epilogue
python3 buffer_versions.py --scenario plain       # 只有 A/B 两个 tile
python3 buffer_versions.py --scenario carry       # 加一个跨迭代存活的 buffer
python3 buffer_versions.py --num-stages 5 --budget-kib 100
```

## 期望输出（本机实际运行结果，`--scenario epilogue`）

```text
num_stages=2  stage=[0, 0, 1]  versions: A_shared=2, B_shared=2, C_shared=1
    no-reuse sum    96.0 KiB | arena    64.0 KiB | budget   228.0 KiB  [OK  ]
      A_shared   offset     0.0 KiB  size    32.0 KiB  live [     3,    51)
      C_shared   offset     0.0 KiB  size    32.0 KiB  live [    51,    53)
      B_shared   offset    32.0 KiB  size    32.0 KiB  live [     4,    51)
num_stages=4  stage=[0, 0, 3]  versions: A_shared=4, B_shared=4, C_shared=1
    no-reuse sum   160.0 KiB | arena   128.0 KiB | budget   228.0 KiB  [OK  ]
      A_shared   offset     0.0 KiB  size    64.0 KiB  live [     9,    57)
      C_shared   offset     0.0 KiB  size    32.0 KiB  live [    57,    59)
      B_shared   offset    64.0 KiB  size    64.0 KiB  live [    10,    57)
```

`--scenario carry`（加一个每次迭代都写、都在最后一 stage 读的 `S_shared`）：

```text
num_stages=4  stage=[0, 0, 0, 3]  versions: A_shared=4, B_shared=4, S_shared=4
    no-reuse sum   192.0 KiB | arena   192.0 KiB | budget   228.0 KiB  [OK  ]
num_stages=5  stage=[0, 0, 0, 4]  versions: A_shared=5, B_shared=5, S_shared=5
    no-reuse sum   240.0 KiB | arena   240.0 KiB | budget   228.0 KiB  [OVER]
```

## 要读出什么

1. **arena 通常小于 no-reuse 之和**：epilogue 场景里 `C_shared` 的活跃区间
   `[51, 53)` 与 A/B 的 `[3, 51)` 不重叠，于是它被放到了 offset 0——
   这就是“哪些 buffer 能复用”的判据：**活跃区间不重叠 + 对齐满足**。
2. **长活跃期的 buffer 会吃掉复用**：carry 场景里 `S_shared` 每次迭代都活着，
   区间覆盖整段循环，任何 tile 都躲不开它，arena 于是等于 no-reuse 之和。
3. **多缓冲与复用是同一笔预算里的两项**：`num_stages` 每加一，A/B 的版本数各加一，
   在 carry 场景下 5 档就顶穿 228 KiB 预算。这对应正文 §3.5 的
   “多缓冲与 buffer 复用冲突”。
4. 对齐（默认 128 B）会让“刚好放得下”变成“刚好放不下”——这就是
   `MergeSharedMemoryAllocations` 里 `align_bytes` 与
   `kSmemAlignmentMap`（TMA/MMA swizzle 的显式对齐）存在的理由。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过，上面引用的输出就是
本机跑出来的。**没有**在任何 GPU/NPU 上对照过。

## 这个模型不做什么

* 它不建模 bank conflict、swizzle、wave quantization，也不建模 fragment/寄存器；
* 它用一个简化的 stage 数组 `[0, 0, N-1]` 代表 `PipelinePlanning` 的结果
  （上游单测对同一形态给出的正是这个数组）；真实 kernel 更复杂的循环体会有别的
  数组，但版本数与 arena 的算法不变；
* 它的 arena 打包与真 pass 是**同一算法、不同实现**，不保证逐字节一致；
  要拿到权威结果，用 `code/01` 在有 TileLang 的机器上打印真实的
  `dynamic_shared_memory` 大小。
