# 图 6：ND 与 NZ 的排布对照

**用途**：正文 §3.4。

| 位置 | 第 1 个 32 B | 第 2 个 32 B | 第 3 个 32 B | 第 4 个 32 B |
| --- | --- | --- | --- | --- |
| **ND（显存）** | token 0 的 head 0–15 | token 0 的 head 16–31 | token 0 的 head 32–47 | token 0 的 head 48–63 |
| **NZ（L1）** | 16 个 token 的 head 0–15 | 16 个 token 的 head 16–31 | …（下一个 fractal） | …（下一个 fractal） |

```text
ND（row-major，内存次序 = 行优先）
  行 0 : [h0..h15][h16..h31][h32..h47][h48..h63] ...
  行 1 : [h0..h15][h16..h31][h32..h47][h48..h63] ...
  ...

NZ（16×16 fractal，内存次序 = fractal 内 head 维优先）
  fractal 0 = 行 0..15 的 head 0..15   （16 行 × 16 列，行内 head 连续）
  fractal 1 = 行 0..15 的 head 16..31
  ...
  再 pad 一行（B_TOPK_PER_V + 1 行）以避免 bank conflict
```

## 读图要点

1. **ND 一行是一个 token 的 512 维**，Cube 读不进；**NZ 一块是 16×16 的 fractal**，
   正好是 L0A/L0B 的输入单元。
2. 两者的「第 k 个 32 B」落在不同的逻辑位置，所以稀疏 gather 无法边搬边转。
3. NZ buffer 在 UB 里多留**一行**（`B_TOPK_PER_V + 1`，`config.h:81`）——
   注释写明是为了在 ND→NZ 写回时避免 UB 的 bank conflict。

## 来源

* 报告 `:146-152`（ND 与 NZ 的定义、随路转换为什么失败）。
* `config.h:81`（`(B_TOPK_PER_V+1)*D_VO` 与注释）；
  `kernel_body.h:287-296`（`block_stride = B_TOPK_PER_V+1` 与两 bank 交错）。
