# 图 3：NVIDIA 侧 scale 在 TMEM 里的布局

用于正文第 4.1 节。这张图说明三件事：scale 放哪里（TMEM 不是 SMEM）、
为什么是“每 4 个 K 块打包”（一个 TMEM 列 32 位 = 4 个 E8M0 字节）、
以及 SFA/SFB 的不对称（SFB 要复制到 32 个 lane 分区）。

```text
SMEM 里（TMA 搬进来的形态）                TMEM 里（MMA 读到的形态）

  一个 word = 4 个字节                      SFA_tmem: [128 lanes, 4 columns]
  ┌────┬────┬────┬────┐                    每个 lane 一行，4 个列 = 4 个 K32 子块
  │ b0 │ b1 │ b2 │ b3 │   b_i = E8M0      ┌────┬────┬────┬────┐
  └────┴────┴────┴────┘   覆盖 gran_k     │ b0 │ b1 │ b2 │ b3 │  lane 0
   K 方向 → 每个字节覆盖一段 K              ├────┼────┼────┼────┤
                                          │ b0 │ b1 │ b2 │ b3 │  lane 1
  128 word 一个 chunk                       ...
  （128 行 × 1 word）                      └────┴────┴────┴────┘
        │
        │ tcgen05_sf_warp_transpose（4x32 -> 32x4）
        ▼
  tcgen05_cp_warpx4：一个 128 word 的 chunk → 4 个 TMEM 列

  MMA 时选哪一列：  SFA_ID = (k_start / gran_k) % 4
                    SFB_ID = (k_start / gran_k) % 4

  ┌───────────────────────────────────────────────────────────────┐
  │ 不对称：PTX 要求 SFB 必须复制到 TMEM 的全部 32 个 lane 分区； │
  │ 而 SFA 在 sm_107f 上若用 tcgen05.st 写入则不需要复制。       │
  │ 后果：SFB 的 TMEM 列数是 SFA 的 2 倍（示例里 4 列 vs 8 列）。│
  └───────────────────────────────────────────────────────────────┘
```

## 为什么是“4 个 K 块打包”

TMEM 的一个单元格是 32 位。E8M0 是 8 位，所以一个单元格正好放 4 个 scale。
PTX 用 `SFA_ID`/`SFB_ID` 这两个 2 位字段在指令描述符里选子列
（`runtime_instr_desc | (sf_a_id << 29) | (sf_b_id << 4)`），
所以“4 个一包”不是工程偏好，而是指令编码决定的。

## 粒度：硬件 32，示例 128

```text
PTX（`.kind::mxf8f6f4` + `.scale_vec::1X`）
    每个 A 的每一行按 32 个 K 元素一个 scale；SFA_ID 选 TMEM word 里的 1 个字节。

锁定版本 SM100 示例（gemm_mxfp8_blockscaled_1d1d.py）
    把粒度**加粗**到 128：同一个字节服务 4 个连续的 K32 原子，
    因为 sf_id = (k_start + ki*32) / 128 在 ki=0..3 上取值相同。
    于是 4 个字节覆盖 4*128 = 512 个 K 元素，正好是一个 word。

昇腾 CATLASS
    MX_SCALE_GROUP_NUM = 32，粒度固定为 32，没有「加粗」这个自由度。
```

## 来源

* `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:32061-32240`（TMEM 布局、
  SFA_ID/SFB_ID、SFB 复制到 32 个 lane 分区）。
* `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/mxfp8_illustrated.md`
  （SFA_tmem `[128 lanes, 4 columns]`、SFB_tmem `[128 lanes, 8 columns]`）。
* `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py:629-632`。
* `_vendor/tilelang-v0.1.15/tilelang/language/builtin.py:1440-1479`。
* `../tilelang-corpus/06-huawei-ascend/raw/catlass/include/catlass/catlass.hpp:48`。
