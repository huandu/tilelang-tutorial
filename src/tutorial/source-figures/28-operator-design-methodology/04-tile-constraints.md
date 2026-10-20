# 图 4：tile 尺寸的四条约束

**用途**：正文第 4.1–4.3 节。

```text
                     block_M, block_N, block_K, stages, threads
                                      |
        +-----------------+-----------+-----------+-----------------+
        |                 |                       |                 |
   C3 指令形状       C2 寄存器压力            C1 片上容量       C4 并行度
   (最便宜，先查)     (最容易忽略)            (最硬的墙)        (最容易被忘)
        |                 |                       |                 |
   mma:  16/8/16     accum/thread           stages*(A+B)      grid/(blk/SM*SMs)
   wgmma:64/8/16     + 地址/索引/流水        <= 可用 shared    -> 波数 + 尾波填充率
   fractal:16x16     <= 255 /线程           <= UB/L1 容量      (真实 SM/核数)
        |                 |                       |                 |
        +-----------------+-----------+-----------+-----------------+
                                      |
                          四条都过 -> 进入候选集合
                          任一条不过 -> 记下是哪一条，回上一步
```

## 4.1 四类约束的“超了会怎样”

| 约束 | NV 侧的硬数字（出处） | 昇腾侧的硬数字（出处） | 超了会怎样 |
| --- | --- | --- | --- |
| 片上容量 | shared 每 SM 228 KB、每 block 227 KB（R7） | L1 示例 512 KB（R14）、UB 示例 192 KB（R13）；L0A/L0B/L0C 容量随型号，需 `GetCoreMemSize` 查询（R12） | 编译或启动阶段直接失败 |
| 寄存器 | 每 SM 65536 个 32 位寄存器、最多 2048 线程、每线程 255 个上限、按每 warp 256 个取整（R8） | 昇腾侧不走“寄存器堆 + occupancy”这条线，走 L0/UB 容量（第 5.4 节） | 溢出到 local memory，性能断崖 |
| 指令形状 | `mma` 16×8×16、`wgmma` 64×N×16、warpgroup=128 线程（R9） | Cube fractal 16×16；对齐要求 UB 32 B、L1 32 B、L0A/L0B 512 B、L0C 64 B（R11） | 编不过，或被迫插入 layout 转换 |
| 并行度 | 输出块数 / (每 SM 驻留块数 × SM 数) | 同上，但网格常是 persistent 的一维固定块数 | 最后一波不满，机器空转 |

## 4.2 “先粗算再微调”的顺序

```text
第 1 步  从「容量允许的最大 tile」开始      —— 因为它同时最小化 DRAM 流量
第 2 步  查指令形状是否整除                —— 不整除就没有意义，先改尺寸
第 3 步  算寄存器与驻留块数                —— 掉到 1 block/SM 就要警惕
第 4 步  算波数与尾波填充率                —— 尾波 < 50% 时优先考虑 split 类变体
第 5 步  只有到这一步才动 block_K / stages —— 它们不影响流量，只影响流水与占用
```

第 5 步的位置很关键：`code/03` 的 carver 复算显示，`block_K` 从 16 到 128 完全不改变
traffic 与 `prio`（前 8 名并列）。**先定 M/N，再定 K**，顺序反了就会在一维无分辨率的
空间里浪费搜索预算。

## 来源

* R7：`nvidia-hopper-tuning-guide.md:50-51,226,228`。
* R8：`nvidia-cuda-best-practices.md:3126`。
* R9：`nvidia-ptx-isa.md:23476-23477`、`:28113-28114`、`:28132`。
* R11：`canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0011_html.md:39-51`（对齐表）。
* R12：`..._10_0008_html.md`（`GetCoreMemSize`）。
* R13 / R14：UB 192 KB 与 L1 512 KB 的**示例条件**，见第 6 章 4.2 节。
* 顺序与并列数量：`code/04-tile-size-decision/`、`code/03-carver-cost-model/` 的实际输出。
