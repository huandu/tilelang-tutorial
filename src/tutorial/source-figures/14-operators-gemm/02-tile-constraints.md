# 图 2：tile 尺寸的两本账

**用途**：正文第 1.3–1.4 节。说明为什么 tile 不能无限开大。

## 一个输出块沿 K 迭代时的资源

```text
        block_N
      ┌─────────┐
      │ C_local │  block_M × block_N × fp32   →  寄存器累加器
      │(fragment)│                             每线程 ≈ (m·n/threads) 个 reg
      └─────────┘
  每个 k 迭代：
      ┌─────────┐   ┌─────────┐
      │ A_tile  │ + │ B_tile  │  × num_stages 份
      │ m × k   │   │ k × n   │  →  shared memory 占用
      └─────────┘   └─────────┘
```

## 三张约束表（NV 视角）

```text
              越接近越好 ←──────────────→ 越接近越危险
 DRAM 流量    b·MNK·(1/m + 1/n)  ≥  b·(MK+KN+MN)
 片上容量     num_stages · b · (m·k + k·n)  ≤  SMEM
 寄存器       (m·n·4) / threads            ≲  255
```

## 一个具体算例（fp16 输入，fp32 累加，threads=128）

```text
 配置                     单级 SMEM   3 级 SMEM   每线程累加寄存器
 128 × 128 × 32           16 KB       48 KB        128
 128 × 256 × 32           24 KB       72 KB        256   ← 已超 255
 256 × 128 × 64           48 KB       96 KB        256   ← 已超 255
 256 × 256 × 32           32 KB       96 KB        512   ← 必然溢出
```

来源：R10（cc7.0 每 SM 65536 个 32 位寄存器、每线程 32 个寄存器才满
occupancy；寄存器分配以每 warp 256 个为单位）、R11（H100 shared memory
可配到 228 KB）、R3（CATLASS 的 L1/L0 约束是同一套账的昇腾版本）。
表中的数值为本书推导，可用 `code/01-gemm-ledger/gemm_ledger.py` 复算。
