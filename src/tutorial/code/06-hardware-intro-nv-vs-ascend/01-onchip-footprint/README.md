# 01 · 片上占用估算器

## 目的

回答本章 1.3 与 4.2 节的那个问题：**给定 tile 形状、dtype 和多缓冲版本数，这块 tile
占多少片上空间？还放得下吗？**

脚本把「逻辑形状 × dtype 字节宽 × 版本数」换算成字节，再和本章 `references.md` 里
有出处的容量对上。它是纯算术，不调用任何硬件、不编译任何 kernel。

## 运行前提

Python 3.9+。只用标准库（`dataclasses`、`argparse`）。**不需要 GPU / NPU / TileLang。**

## 运行命令

```bash
cd 01-onchip-footprint
python3 onchip_footprint.py
```

## 输出解读

脚本跑三个 case：

* **Case 1（NV）**：`A_shared(128,64) + B_shared(64,128)`，fp16，开 3 份多缓冲 =
  96 KiB。对照 H100 单 block 的 227 KB 上限，还剩 131 KiB。
* **Case 2（昇腾 L1）**：`A_l1(256,128) + B_l1(224,128)`，bf16，2 份多缓冲 =
  240 KiB。对照文档示例里的 512 KB L1，还剩 272 KiB。
* **Case 3（昇腾 UB）**：一个 `C_ub(128,256)` fp32 tile，1 份就要 128 KiB，2 份
  要 256 KiB——超过文档示例里 192 K 的 UB 上限。这个 case 的作用正是说明：**多开一份
  缓冲不是免费的**，它和 tile 形状、dtype 三者互相挤压。

## 期望输出（摘要）

在 macOS（Apple Silicon）、Python 3.11 上实际运行得到的摘要：

```text
Case 1: NV GEMM A/B tiles (block_M=block_N=128, block_K=64, fp16)
  A_shared (128, 64)          fp16  1 copy =     16,384 B (16 KiB)
  B_shared (64, 128)          fp16  1 copy =     16,384 B (16 KiB)
3 version(s) total  :       98,304 B (96 KiB)
  [   FITS] NV shared memory per block (H100, cc9.0)     131 KiB +

Case 2: Ascend GEMM A/B tiles in L1 (block_M=256, block_N=224, block_K=128, bf16)
2 version(s) total  :      245,760 B (240 KiB)
  [   FITS] Ascend L1 Buffer (documented example)        272 KiB +

Case 3: Ascend vector tiles in UB (a 128x256 fp32 tile, 2 versions)
2 version(s) total  :      262,144 B (256 KiB)
  [TOO BIG] Ascend Unified Buffer (documented example)   64 KiB -
```

## 重要边界

* 脚本里 H100 / B200 的容量是**厂商公开的每 SM / 每 block 上限**（来源见
  `../../references.md` 的 R2、R4），不是实测。
* 脚本里昇腾 UB=192 K、L1=512 KB 都是**某份官方文档里的示例数字**，不是昇腾家族
  通用规格；L0A/L0B/L0C 的容量故意没有放进 catalogue，因为本章没有找到可跨型号引用
  的数字（应当用 `GetCoreMemSize` 在真实设备上查询）。
* 脚本检查的是「字节数放不放得下」，**不检查**对齐、bank 冲突、swizzle、layout
  约束——这些要等第 10、21 章。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过，上面的输出即实际输出。
**未在任何加速器上运行，也未与任何真实编译结果对照。**
